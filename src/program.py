import asyncio
import websockets
import time
import json 
import os

import signal
import string
import random
import cv2
import threading
import queue
import requests
import numpy as np

# from configure import update_env_variables,restart_service
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# Global variables
camera_threads = {}  # Dictionary to store camera threads
frame_queues = {}   # Dictionary to store frame queues for each camera
max_queue_size = 30
max_processing_threads = 3
index = 0

capture_interval_seconds = int(os.getenv("CAPTURE_INTERVAL_SECONDS", 3))
frame_no_change_interval = int(os.getenv("FRAME_NO_CHANGE_INTERVAL", 20))
weapon_detection_url = os.getenv("WEAPON_DETECTION_URL", "http://192.168.1.52:8080/predictions/accelx-weapon-dt-yolos-detr")
detect_image_save = os.getenv("DETECT_IMAGE_SAVE", "http://192.168.1.157:8001/camera/weapon-detect-images/")
camera_setups_url = os.getenv("CAMERA_SETUP_URL", "http://192.168.1.157:8001/camera/camera-setups/")
cam_check_url_save = os.getenv("CAM_URL_CHECK", "http://192.168.1.157:8001/camera/checkcam-url-images/")
group_name = os.getenv("GROUP_NAME")
secret_key = os.getenv("SECRET_KEY")


print(os.getenv("URI"))


class Frame:
    _instance = None  # Class variable to store the singleton instance
    
    @staticmethod
    def get_instance():
        if Frame._instance is None:
            Frame._instance = Frame()
        return Frame._instance
    
    def __init__(self):        
        self.reference_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.last_output_time = time.time()
    def compare_frame(self, target_frame, threshold=8):
        
        if(self.reference_frame.shape != target_frame.shape):
            height, width, channels = target_frame.shape
            self.reference_frame = np.zeros((height,width,channels), dtype=np.uint8)
            
        # Convert -> Gray Scale
        reference_gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Absolute Difference
        frame_diff = cv2.absdiff(reference_gray, target_gray)
        
        # Calculate Mean Of Absolute Difference
        mean_diff = frame_diff.mean()
        # print('mean_diff ',mean_diff)
        
        current_time = time.time()
        # print('current_time ',current_time)
        # print('last_output_time ',self.last_output_time)
        elapsed_time = current_time - self.last_output_time
        # print('elapsed_time ',elapsed_time)

        if mean_diff > threshold or elapsed_time >=frame_no_change_interval:
            self.last_output_time = current_time
            self.reference_frame = target_frame
            return True

        return None


class CameraThreadInfo:
    def __init__(self):
        self.capture_thread = None
        self.process_threads = []
        self.last_process_time = time.time()
        

class StoppableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self.stopped = threading.Event()

    def stop(self):
        self.stopped.set()


class SignalHandler:
    shutdown_requested = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.request_shutdown)
        signal.signal(signal.SIGTERM, self.request_shutdown)
    
    def request_shutdown(self, *args):
        print('Request to shutdown received, stopping')
        self.shutdown_requested = True
        
    def can_run(self):
        return not self.shutdown_requested


        
def generate_random_text(length):
    characters = string.ascii_letters + string.digits
    random_text = ''.join(random.choice(characters) for _ in range(length))
    return random_text


def open_cam_rtsp(uri, width, height, latency):
    print('uri ',uri)
    #frame_rate = capture_interval_seconds
    gst_str = ("rtspsrc location={} latency={} ! queue ! rtph264depay ! h264parse ! omxh264dec ! "
               "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
               "videoconvert ! videorate ! video/x-raw, framerate=1/{} ! appsink").format(uri, latency, width, height, capture_interval_seconds)
    
    try:
        # Attempt to open the camera with the GStreamer pipeline
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        return cap
    except cv2.error as e:
        # Handle GStreamer errors
        print("Error: Unable to open camera with GStreamer pipeline:", e)
        return None


def get_frame_rtsp(uri, width, height, latency):
    print('cam url ',uri)
    gst_str = ("rtspsrc location={} latency={} ! queue ! rtph264depay ! h264parse ! omxh264dec ! "
               "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
               "videoconvert ! appsink").format(uri, latency, width, height)
    
    try:
        # Attempt to open the camera with the GStreamer pipeline
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        return cap
    except cv2.error as e:
        # Handle GStreamer errors
        print("Error: Unable to open camera with GStreamer pipeline:", e)
        return None


def capture_frames(camera_url, cam_id, cam_type):
    global frame_queues, signal_handler
    width = int(os.getenv("WIDTH", 1080))
    height = int(os.getenv("HEIGHT", 720))
    latency = int(os.getenv("LATENCY", 100))
    
    
    if cam_type == 'jpeg':  # Check if the URL is HTTP/HTTPS
        while cam_id in camera_threads and not (signal_handler.shutdown_requested or camera_threads[cam_id].capture_thread.stopped.is_set()):
            try:
                # Capture the image from the URL
                response = requests.get(camera_url)
                if response.status_code == 200:
                    frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

                    frame_queue = frame_queues[cam_id]
                    if frame_queue.qsize() < max_queue_size:
                        absdiff_check = Frame.get_instance()
                        accurate = absdiff_check.compare_frame(target_frame=frame, threshold=int(os.getenv("THRESHOLD", 8)))
                        if accurate:
                            frame_queue.put(frame)
                else:
                    print(f"Failed to fetch image from URL {camera_url}. Status code: {response.status_code}")
                    no_signal_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    text_size = cv2.getTextSize("No Signal", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (width - text_size[0]) // 2
                    text_y = (height + text_size[1]) // 2
                    cv2.putText(no_signal_frame, "No Signal", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    print('done')
                    no_frame = cv2.imencode('.jpg', no_signal_frame)[1].tobytes()
                    files = {'detect_image': (f'frame_{cam_id}.jpg', no_frame, 'image/jpeg')}
                    data = {'camera': cam_id, 'detect_event': 'No Signal'}
                    response = requests.post(detect_image_save, files=files, data=data)
                    patch_data = {"camera_frame_cap_status": False}
                    api_url = f'{camera_setups_url}{cam_id}/'
                    #print('api_url ',api_url)
                    requests.patch(api_url, json=patch_data)
                    time.sleep(10)
                    continue
                
                time.sleep(capture_interval_seconds)
            except Exception as e:
                print(f"Error capturing frame: {e}")
    
    elif cam_type == 'rtsp':
        cap = open_cam_rtsp(camera_url, width, height, latency)
        #frameRate = int(cap.get(cv2.CAP_PROP_FPS))
        
        if cap is None:
            print("Error: Failed to open camera with GStreamer pipeline.")
            # Perform error handling actions, such as logging or graceful shutdown
        else:
        
            #print(f"capture_interval_seconds: {capture_interval_seconds}")
            
            n = 0

            while cam_id in camera_threads and not (signal_handler.shutdown_requested or camera_threads[cam_id].capture_thread.stopped.is_set()):
                if not cap.isOpened():
                    print(f"Error: Unable to open camera for URI {camera_url}")

                    no_signal_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    text_size = cv2.getTextSize("No Signal", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (width - text_size[0]) // 2
                    text_y = (height + text_size[1]) // 2
                    cv2.putText(no_signal_frame, "No Signal", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    print('done')
                    no_frame = cv2.imencode('.jpg', no_signal_frame)[1].tobytes()
                    files = {'detect_image': (f'frame_{cam_id}.jpg', no_frame, 'image/jpeg')}

                    data = {'camera': cam_id, 'detect_event': 'No Signal'}
                    response = requests.post(detect_image_save, files=files, data=data)
                    patch_data = {"camera_frame_cap_status": False}
                    api_url = f'{camera_setups_url}{cam_id}/'
                    #print('api_url ',api_url)
                    requests.patch(api_url, json=patch_data)
                    time.sleep(10)
                    continue
                
                ret, frame = cap.read()
                
                if not ret:

                    break

                # if n % int(frameRate * capture_interval_seconds) == 0:
                    #print('yes')
                    
                frame_queue = frame_queues[cam_id]
                if frame_queue.qsize() < max_queue_size:
                    print('yes ')
                    absdiff_check = Frame.get_instance()
                    accurate = absdiff_check.compare_frame(target_frame=frame,threshold=int(os.getenv("THRESHOLD", 8)))
                    if accurate:
                        print('done')
                        frame_queue.put(frame)
                        
                # start_time = time.time()
                # print('start_time ',start_time)
                
                
                # time.sleep(3)
                # end_time = time.time()
                # interval = end_time - start_time
                # print('interval ',interval)
                # print('time working')
                n += 1

                

            cap.release()


def process_frames(thread_id, cam_id):
    global frame_queues, max_processing_threads, signal_handler

    while cam_id in camera_threads and not (signal_handler.shutdown_requested or camera_threads[cam_id].capture_thread.stopped.is_set()):
        # Check if cam_id exists in frame_queues
        if cam_id not in frame_queues:
            print(f"Thread {thread_id} for Camera {cam_id}: Frame queue not found.")
            continue

        frame_queue = frame_queues[cam_id]
        
        try:
            # Wait until the queue has frames
            frame = frame_queue.get(timeout=1)
            
            # Do some processing on the frame (replace this with your actual processing logic)
            processed_frame = cv2.putText(frame, f"Camera {cam_id}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            weapon_image = cv2.imencode('.jpg', processed_frame)[1].tobytes()


            res = requests.post(weapon_detection_url, data=weapon_image)
            detect = json.loads(res.content)

            
            # Check if there are probabilities and bounding boxes
            if 'probs' in detect and 'bboxes_scaled' in detect:
                bboxes_list = detect['bboxes_scaled']
                #print('bboxes_list', bboxes_list)
                probs_list = detect['probs']
                weapon = detect['id2label']['0']

                # Check if bboxes_list is not empty
                if probs_list and bboxes_list:
                    for probs, bbox_scaled in zip(probs_list, bboxes_list):
                        #print(f'cam {bboxes_list}') 
                        # Check the condition for bbox_scaled value
                        if isinstance(bbox_scaled, (list, tuple)) and len(bbox_scaled) == 4:
                            x_min, y_min, x_max, y_max = bbox_scaled

                            # Convert bbox_scaled to absolute pixel values
                            image_width, image_height = processed_frame.shape[1], processed_frame.shape[0]
                            x_pixel = int(x_min)
                            y_pixel = int(y_min)
                            width_pixel = int((x_max - x_min))
                            height_pixel = int((y_max - y_min))

                            # Create a bounding box
                            bbox = (x_pixel, y_pixel, width_pixel, height_pixel)
                            #print('bbox', bbox)
                            prob = probs[0]

                            # Draw the bounding box on the image
                            cv2.rectangle(processed_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
                            text = f'{weapon}: {prob:.2f}'
                            
                            cv2.putText(processed_frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (249, 246, 246), 2)
                    
                    weapon_image = cv2.imencode('.jpg', processed_frame)[1].tobytes()
                    # Define the API endpoint
                    #api_url = "http://192.168.1.157:8001/camera/weapon-detect-images/"

                    random_text = generate_random_text(6)
                    # Make the POST request with the encoded image
                    files = {'detect_image': (f'frame_{random_text}.jpg', weapon_image, 'image/jpeg')}
                    #logging.info(f"type came id: {type(camera_id)}")
                    data = {'camera': cam_id, 'detect_event': weapon}
                    response = requests.post(detect_image_save, files=files, data=data)

                    print(f"Thread {thread_id} for Camera {cam_id}: Frame processed with Event detect.")
                
                if not bboxes_list:
                    files = {'detect_image': (f'frame_{cam_id}.jpg', weapon_image, 'image/jpeg')}
                    #logging.info(f"type came id: {type(camera_id)}")
                    data = {'camera': cam_id, 'detect_event': 'No Event'}
                    response = requests.post(detect_image_save, files=files, data=data)

                    print(f"Thread {thread_id} for Camera {cam_id}: Frame processed with NO Event detect.")

            

        except queue.Empty:
            # Queue is empty, no frames to process
            print(f"Camera {cam_id} Queue is empty")
        except Exception as e:
            print(f"Error processing frame: {e} for Camera {cam_id}")


def create_processing_thread(thread_id, cam_id):
    new_thread = threading.Thread(target=process_frames, args=(thread_id, cam_id))
    new_thread.start()
    return new_thread


def set_config_file(camera_url, camera_id, camera_type, camera_running_status, filename='camera_config.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
            found = False
            for entry in data:
                if entry["camera_id"] == camera_id:
                    entry["camera_url"] = camera_url
                    entry["camera_type"] = camera_type
                    entry["camera_running_status"] = camera_running_status
                    found = True
                    break
            if not found:
                data.append({"camera_id": camera_id, "camera_url": camera_url, "camera_type": camera_type, "camera_running_status": camera_running_status})

        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    else:
        data = [{"camera_id": camera_id, "camera_url": camera_url, "camera_type": camera_type, "camera_running_status": camera_running_status}]
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)



async def start_capture_async(cam_url, cam_id, cam_type):
    try:
        if cam_id in camera_threads:
            raise ValueError(f"Capture for Camera ID {cam_id} is already running.")

        # Initialize thread info for the camera
        camera_threads[cam_id] = CameraThreadInfo()

        # Initialize frame queue for the camera
        frame_queues[cam_id] = queue.Queue(maxsize=max_queue_size)

        # Create and start capture thread
        camera_threads[cam_id].capture_thread = StoppableThread(target=capture_frames, args=(cam_url, cam_id, cam_type))
        camera_threads[cam_id].capture_thread.start()

        # Create initial processing threads after capture thread has started
        camera_threads[cam_id].process_threads = [
            create_processing_thread(i, cam_id) for i in range(min(max_processing_threads, max_queue_size))
        ]

        return {"message": f"Capture started for Camera ID {cam_id} with URL {cam_url}"}
    except Exception as e:
        return {"error": str(e)}


async def stop_capture_async(cam_id):
    global camera_threads, frame_queues

    try:
        if cam_id not in camera_threads:
            raise ValueError(f"No capture is running for Camera ID {cam_id}.")

        # Stop capture thread
        camera_threads[cam_id].capture_thread.stop()
        camera_threads[cam_id].capture_thread.join()

        # Stop processing threads
        for process_thread in camera_threads[cam_id].process_threads:
            process_thread.join()

        # Remove camera info
        del camera_threads[cam_id]
        del frame_queues[cam_id]

        return {"message": f"Capture stopped for Camera ID {cam_id}"}
    except Exception as e:
        return {"error": str(e)}


async def cam_check_async(cam_url, cam_type):
    try:
        width = 1080
        height = 720
        latency = 100
        
        if cam_type == 'jpeg':
            response = requests.get(cam_url)
            if response.status_code == 200:
                frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                
                random_text = generate_random_text(6)
                files = {'image': (f'frame_{random_text}.jpg', frame, 'image/jpeg')}
                data = {'is_readable': True}

                response = requests.post(cam_check_url_save, files=files, data=data)

                if response.status_code == 201:
                    res_data = json.loads(response.content)
                    check_image_id = res_data['id']
                    return {"check_cam_url": check_image_id , "group_name": group_name ,"status_code": 2001}
            else:
                return {"message": f"frame not Captured with URL {cam_url}", "status_code": 2005, "cam_url": cam_url}
        
        elif cam_type == 'rtsp':
            cap = get_frame_rtsp(cam_url, width, height, latency)
            if not cap.isOpened():
                return {"message": f"frame not Captured with URL {cam_url}", "status_code": 2005, "cam_url": cam_url}
            
            success, frame = cap.read()
            if not success:
                return {"message": f"frame not Captured with URL {cam_url}", "status_code": 2005, "cam_url": cam_url}
            
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            random_text = generate_random_text(6)
            files = {'image': (f'frame_{random_text}.jpg', frame, 'image/jpeg')}
            data = {'is_readable': True}

            response = requests.post(cam_check_url_save, files=files, data=data)
            
            if response.status_code == 201:
                res_data = json.loads(response.content)
                check_image_id = res_data['id']
                return {"check_cam_url": check_image_id , "group_name": group_name ,"status_code": 2001}
            
        else:
            return {"message": "Unsupported camera URL format", "status_code": 400}
        
    except Exception as e:
        return {"error": str(e)}


async def connect_to_websocket():
    global flag
    uri = os.getenv("URI")
    while signal_handler.can_run():
        try:
            async with websockets.connect(uri) as websocket:
                
                print("Connected to WebSocket server")
                
                if os.path.exists('camera_config.json'):
                    with open('camera_config.json', 'r') as json_file:
                        data = json.load(json_file)
                    for entry in data:
                        if entry['camera_running_status'] == True:
                            camera_id = entry['camera_id']
                            response = await start_capture_async(entry['camera_url'], entry['camera_id'], entry['camera_type'])
                            response_json = json.dumps(response)
                            
                            await websocket.send(response_json)
                            patch_data = {"camera_running_status": True, "camera_frame_cap_status": True}
                            api_url = f'{camera_setups_url}{camera_id}/'
                            print(api_url)
                            requests.patch(api_url, json=patch_data)
                    print("First Time Calling Start_Capture_Async")
                
                message_to_send = {"message": f"message from {group_name} server", "status_code": 2000, "video_process_server_id": group_name}
                message_json = json.dumps(message_to_send)
                await websocket.send(message_json)
                print(f"Sent message: {message_json}")

                # Define a task for sending heartbeat messages
                async def send_heartbeat():
                    while signal_handler.can_run():
                        await asyncio.sleep(10)  # Adjust the interval as needed
                        try:
                            #await websocket.ping()
                            await websocket.send(message_json)
                            print("send heartbeat")
                        except websockets.ConnectionClosed:
                            print("Connection closed while sending heartbeat.")
                            break

                heartbeat_task = asyncio.ensure_future(send_heartbeat())

                try:
                    while signal_handler.can_run():
                        message_received = await websocket.recv()
                        print('message_received ', message_received)

                        try:
                            data = json.loads(message_received)
                            print(f"Received message: {message_received}")
                            if 'camera_running_status' in data and data['camera_running_status']:
                                # Call the start_capture_async function
                                print('going to start_capture_async')
                                set_config_file(data.get('camera_url'), data.get('camera_id'),data.get('camera_type'),data.get('camera_running_status'))
                                response = await start_capture_async(data.get('camera_url'), data.get('camera_id'), data.get('camera_type'))
                                response_json = json.dumps(response)
                                await websocket.send(response_json)
                            elif 'camera_running_status' in data and not data['camera_running_status']:
                                # Call the stop_capture_async function
                                set_config_file(data.get('camera_url'), data.get('camera_id'),data.get('camera_type'),data.get('camera_running_status'))
                                response = await stop_capture_async(data.get('camera_id'))
                                response_json = json.dumps(response)
                                await websocket.send(response_json)
                            elif 'camera_check_url' in data:
                                print('camera_check_url ',data.get('camera_check_url'))
            
                                response = await cam_check_async(data.get('camera_check_url'), data.get('camera_type'))
                                print('response ',response)
                                response_json = json.dumps(response)
                                await websocket.send(response_json)
                            # elif 'change_in_env' in data:
                            #     response = await update_env_variables()
                        except:
                            print("Error processing received message.")
                            continue
                        
                        pass

                except websockets.ConnectionClosed:
                    print("Connection closed by server.")

                finally:
                    # Cancel the heartbeat task when the main loop ends
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            print(f"Connection error: {e}")
            print("Reconnecting in 5 seconds...")
            await asyncio.sleep(1)


signal_handler = SignalHandler()

loop = asyncio.get_event_loop()
loop.run_until_complete(connect_to_websocket())