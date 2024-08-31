import asyncio
import uuid
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
import logging

from dotenv import load_dotenv



#Thread Settings
camera_threads = {}  
frame_queues = {}   
max_queue_size = 30
max_processing_threads = 3
index = 0

#.env Settings
load_dotenv()
group_name = os.getenv("SERVER_ID")
secret_key = os.getenv("SECRET_KEY")
base_uri = os.getenv("URI")
if(base_uri.startswith("http://")):
    socket_uri = base_uri.replace("http://", "ws://")
else:
    socket_uri = base_uri.replace("https://", "wss://")   

connection_uri = socket_uri + "ws/video-process/" + group_name + "/"
detect_image_save = base_uri + "camera/weapon-detect-images/"
camera_setups_url = base_uri + "camera/camera-setups/"
cam_check_url_save = base_uri + "camera/checkcam-url-images/"

capture_interval_seconds = int(os.getenv("CAPTURE_INTERVAL_SECONDS", 3))
frame_no_change_interval = int(os.getenv("FRAME_NO_CHANGE_INTERVAL", 20))
weapon_detection_url = os.getenv("WEAPON_DETECTION_URL", "http://192.168.1.52:8080/predictions/accelx-weapon-dt-yolos-detr")

#Log
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)




#Initiate Static Frame
class Frame:
    _instance = None  

    @staticmethod
    def get_instance():
        if Frame._instance is None:
            Frame._instance = Frame()
        return Frame._instance

    def __init__(self):        
        self.reference_frame = {}
        self.last_output_time = {}
    def compare_frame(self,cam_id, target_frame, threshold=8.0):

        if cam_id not in self.reference_frame or self.reference_frame[cam_id].shape != target_frame.shape:
            height, width, channels = target_frame.shape
            self.reference_frame[cam_id] = np.zeros((height,width,channels), dtype=np.uint8)

        reference_gray = cv2.cvtColor(self.reference_frame[cam_id], cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(reference_gray, target_gray)

        mean_diff = frame_diff.mean()
        print(f'mean_diff: {mean_diff} cam_id: {cam_id}')

        if cam_id not in self.last_output_time:
            self.last_output_time[cam_id]= time.time()

        current_time = time.time()

        elapsed_time = current_time - self.last_output_time[cam_id]

        print(f'elapsed_time: {elapsed_time} cam_id: {cam_id}')

        if mean_diff > threshold or elapsed_time >=frame_no_change_interval:
            # print(f'Cam id {cam_id} is being sent for enqueue')
            self.last_output_time[cam_id] = current_time
            self.reference_frame[cam_id] = target_frame
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
    
    uri = uri.replace("rtsp://", "rtspt://")
    print('uri ',uri)
    gst_str = ( 'rtspsrc async-handling=true location={} latency={} retry=5 !'
                'rtph264depay ! h264parse !'
                'queue max-size-buffers=100 leaky=2 !'
                'nvv4l2decoder enable-max-performance=1 !'
                'video/x-raw(memory:NVMM), format=(string)NV12 !'
                'nvvidconv ! video/x-raw, width={}, height={}, format=(string)BGRx !'
                'videorate ! video/x-raw, framerate=(fraction)1/{} !'
                'videoconvert ! '
                'appsink').format(uri, latency, width, height, capture_interval_seconds)    

    try:
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("Failed to open video stream")
        else:
            return cap
    except cv2.error as e:
        print("Error: Unable to open camera with GStreamer pipeline:", e)
        return None
    except Exception as e:
        print("An error occurred during video capture:", e)
        return None

def get_frame_rtsp(uri, width, height, latency):
    
    uri = uri.replace("rtsp://", "rtspt://")
    print('cam url ',uri)
    gst_str = ( 'rtspsrc async-handling=true location={} latency={} retry=5 !'
                'rtph264depay ! h264parse !'
                'queue max-size-buffers=100 leaky=2 !'
                'nvv4l2decoder enable-max-performance=1 !'
                'video/x-raw(memory:NVMM), format=(string)NV12 !'
                'nvvidconv ! video/x-raw, width={}, height={}, format=(string)BGRx !'
                'videorate ! video/x-raw, framerate=(fraction)1/{} !'
                'videoconvert ! '
                'appsink').format(uri, latency, width, height, capture_interval_seconds)   

    try:

        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        return cap
    except cv2.error as e:

        print("Error: Unable to open camera with GStreamer pipeline:", e)
        return None

def capture_frames(camera_url, cam_id, cam_type, threshold=float(os.getenv("THRESHOLD", 8.0))):
    global frame_queues, signal_handler
    width = int(os.getenv("WIDTH", 1080))
    height = int(os.getenv("HEIGHT", 720))
    latency = int(os.getenv("LATENCY", 100))
    n = {}

    if cam_type == 'jpeg':  
        while cam_id in camera_threads and not (signal_handler.shutdown_requested or camera_threads[cam_id].capture_thread.stopped.is_set()):
            try:

                response = requests.get(camera_url)
                if response.status_code == 200:
                    frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

                    frame_queue = frame_queues[cam_id]
                    if frame_queue.qsize() < max_queue_size:
                        absdiff_check = Frame.get_instance()
                        accurate = absdiff_check.compare_frame(cam_id=cam_id, target_frame=frame, threshold=threshold)
                        if accurate:
                            frame_queue.put(frame)
                else:
                    print(f"Failed to fetch image from URL {camera_url}. Status code: {response.status_code}")
                    no_signal_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    text_size = cv2.getTextSize("No Signal", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (width - text_size[0]) // 2
                    text_y = (height + text_size[1]) // 2
                    cv2.putText(no_signal_frame, "No Signal", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(no_signal_frame, f"Camera {cam_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    no_frame = cv2.imencode('.jpg', no_signal_frame)[1].tobytes()
                    files = {'detect_image': (f'frame_{cam_id}.jpg', no_frame, 'image/jpeg')}
                    data = {'camera': cam_id, 'detect_event': 'No Signal'}
                    response = requests.post(detect_image_save, files=files, data=data)
                    patch_data = {"camera_frame_cap_status": False}
                    api_url = f'{camera_setups_url}{cam_id}/'

                    requests.patch(api_url, json=patch_data)

                    time.sleep(10)
                    continue

                time.sleep(capture_interval_seconds)
            except Exception as e:
                print(f"Error capturing frame: {e}")

    elif cam_type == 'rtsp':
        cap = open_cam_rtsp(camera_url, width, height, latency)
        
        while cam_id in camera_threads and not (signal_handler.shutdown_requested or camera_threads[cam_id].capture_thread.stopped.is_set()):
            # print(f"camera id: {cam_id}, cap: {cap}")
            if cap is None or not cap.isOpened():
                print(f"Error: Unable to open camera for URI {camera_url}")
                if cap is not None:
                    cap.release()
                no_signal_frame = np.zeros((height, width, 3), dtype=np.uint8)
                text_size = cv2.getTextSize("No Signal", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (width - text_size[0]) // 2
                text_y = (height + text_size[1]) // 2
                cv2.putText(no_signal_frame, "No Signal", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(no_signal_frame, f"Camera {cam_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                no_frame = cv2.imencode('.jpg', no_signal_frame)[1].tobytes()
                files = {'detect_image': (f'frame_{cam_id}.jpg', no_frame, 'image/jpeg')}

                data = {'camera': cam_id, 'detect_event': 'No Signal'}
                response = requests.post(detect_image_save, files=files, data=data)
                patch_data = {"camera_frame_cap_status": False}
                api_url = f'{camera_setups_url}{cam_id}/'

                requests.patch(api_url, json=patch_data)
                time.sleep(10)
                cap = open_cam_rtsp(camera_url, width, height, latency)
                continue

            ret, frame = cap.read()
            
            
            if not ret:
                cap.release()
                time.sleep(1)
                cap = open_cam_rtsp(camera_url, width, height, latency)
                continue

            frame_queue = frame_queues[cam_id]
            if frame_queue.qsize() < max_queue_size:

                absdiff_check = Frame.get_instance()
                accurate = absdiff_check.compare_frame(cam_id=cam_id, target_frame=frame,threshold=threshold)
                if accurate:
                    # print(f"Cam ID: {cam_id}! Ready For Processing")
                    frame_queue.put(frame)
                    continue
                
            

        if cap is not None:
            print("cap is released")
            print(camera_threads[cam_id],camera_threads[cam_id].capture_thread.stopped.is_set())
            cap.release()

def process_frames(thread_id, cam_id):
    global frame_queues, max_processing_threads, signal_handler

    while cam_id in camera_threads and not (signal_handler.shutdown_requested or camera_threads[cam_id].capture_thread.stopped.is_set()):

        if cam_id not in frame_queues:
            print(f"Thread {thread_id} for Camera {cam_id}: Frame queue not found.")
            continue

        frame_queue = frame_queues[cam_id]

        try:
            
            frame = frame_queue.get(timeout=1)
            
            processed_frame = cv2.putText(frame, f"Camera {cam_id}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            weapon_image = cv2.imencode('.jpg', processed_frame)[1].tobytes()
            # print(f"Cam ID: {cam_id} Starting To Process")
            res = requests.post(weapon_detection_url, data=weapon_image)
            detect = json.loads(res.content)

            if 'probs' in detect and 'bboxes_scaled' in detect:
                bboxes_list = detect['bboxes_scaled']

                probs_list = detect['probs']
                weapon = detect['id2label']['0']

                if probs_list and bboxes_list:
                    for probs, bbox_scaled in zip(probs_list, bboxes_list):

                        if isinstance(bbox_scaled, (list, tuple)) and len(bbox_scaled) == 4:
                            x_min, y_min, x_max, y_max = bbox_scaled

                            image_width, image_height = processed_frame.shape[1], processed_frame.shape[0]
                            x_pixel = int(x_min)
                            y_pixel = int(y_min)
                            width_pixel = int((x_max - x_min))
                            height_pixel = int((y_max - y_min))

                            bbox = (x_pixel, y_pixel, width_pixel, height_pixel)

                            prob = probs[0]

                            cv2.rectangle(processed_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
                            text = f'{weapon}: {prob:.2f}'

                            cv2.putText(processed_frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (249, 246, 246), 2)

                    weapon_image = cv2.imencode('.jpg', processed_frame)[1].tobytes()

                    random_text = generate_random_text(6)

                    files = {'detect_image': (f'frame_{random_text}.jpg', weapon_image, 'image/jpeg')}

                    data = {'camera': cam_id, 'detect_event': weapon}
                    response = requests.post(detect_image_save, files=files, data=data)

                    print(f"Thread {thread_id} for Camera {cam_id}: Frame processed with Event detect.")

                if not bboxes_list:
                    files = {'detect_image': (f'frame_{cam_id}.jpg', weapon_image, 'image/jpeg')}

                    data = {'camera': cam_id, 'detect_event': 'No Event'}
                    response = requests.post(detect_image_save, files=files, data=data)

                    print(f"Thread {thread_id} for Camera {cam_id}: Frame processed with NO Event detect.")

        except queue.Empty:

            pass
        except Exception as e:
            print(f"Error processing frame: {e} for Camera {cam_id}")

def create_processing_thread(thread_id, cam_id):
    new_thread = threading.Thread(target=process_frames, args=(thread_id, cam_id))
    new_thread.start()
    return new_thread

def set_config_file(camera_url, camera_id, camera_type, camera_running_status, threshold, filename='camera_config.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
            found = False
            for entry in data:
                if entry["camera_id"] == camera_id:
                    entry["camera_url"] = camera_url
                    entry["camera_type"] = camera_type
                    entry["camera_running_status"] = camera_running_status
                    entry["threshold"] = threshold
                    found = True
                    break
            if not found:
                data.append({"camera_id": camera_id, "camera_url": camera_url, "camera_type": camera_type, "camera_running_status": camera_running_status, "threshold": threshold})

        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    else:
        data = [{"camera_id": camera_id, "camera_url": camera_url, "camera_type": camera_type, "camera_running_status": camera_running_status, "threshold": threshold}]
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)

def unset_config_file(camera_id, filename='camera_config.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
            updated_data = [entry for entry in data if entry["camera_id"] != camera_id]

        with open(filename, 'w') as json_file:
            json.dump(updated_data, json_file, indent=4)
    else:
        print("Config file not found.")

async def start_capture_async(cam_url, cam_id, cam_type, threshold):
    try:
        if cam_id in camera_threads:
            raise ValueError(f"Capture for Camera ID {cam_id} is already running.")

        camera_threads[cam_id] = CameraThreadInfo()

        frame_queues[cam_id] = queue.Queue(maxsize=max_queue_size)

        camera_threads[cam_id].capture_thread = StoppableThread(target=capture_frames, args=(cam_url, cam_id, cam_type, threshold))
        camera_threads[cam_id].capture_thread.start()

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

        camera_threads[cam_id].capture_thread.stop()
        camera_threads[cam_id].capture_thread.join()

        for process_thread in camera_threads[cam_id].process_threads:
            process_thread.join()

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
                data = {'is_readable': True, 'video_server_id': group_name}

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
                cap.release()
                return {"message": f"frame not Captured with URL {cam_url}", "status_code": 2005, "cam_url": cam_url}

            success, frame = cap.read()
            if not success:
                cap.release()
                return {"message": f"frame not Captured with URL {cam_url}", "status_code": 2005, "cam_url": cam_url}
            cap.release()
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            random_text = generate_random_text(6)
            files = {'image': (f'frame_{random_text}.jpg', frame, 'image/jpeg')}
            data = {'is_readable': True, 'video_server_id': group_name}

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
    uri = connection_uri
    while signal_handler.can_run():
        try:
            async with websockets.connect(uri) as websocket:

                print("Connected to WebSocket server")

                if os.path.exists('camera_config.json'):
                    with open('camera_config.json', 'r') as json_file:
                        data = json.load(json_file)
                    for entry in data:                        
                        camera_id = entry['camera_id']
                        api_url = f'{camera_setups_url}{camera_id}/'
                        cam_found = requests.get(api_url)
                        cam_info = cam_found.json()

                        if cam_found.status_code == 200:
                            video_process_server_id = cam_info['video_process_server_info']['video_process_server_id']
                            if video_process_server_id == group_name:  
                                if entry['camera_running_status'] == True:
                                    response = await start_capture_async(entry['camera_url'], entry['camera_id'], entry['camera_type'], entry['threshold'])
                                    response_json = json.dumps(response)

                                    await websocket.send(response_json)
                                    patch_data = {"camera_running_status": True, "camera_frame_cap_status": True}

                                    print(api_url)
                                    resf = requests.patch(api_url, json=patch_data)
                                    print(f"Patch Data")
                                    if resf.status_code == 200:
                                        print("Success")
                                        pass
                                    else:
                                        await stop_capture_async(entry['camera_id'])
                                        print(f"Camera {entry['camera_id']} didn't start")    
                            else:
                                unset_config_file(camera_id)

                                print(f"Deleted Camera: {camera_id}!")

                        else:
                            unset_config_file(camera_id)

                            print(f"Deleted Camera: {camera_id}!")
                        time.sleep(2)
                    print("First Time Calling Start_Capture_Async")


                message_to_send = {"message": f"message from {group_name} server", "status_code": 2000, "video_process_server_id": group_name}
                message_json = json.dumps(message_to_send)
                await websocket.send(message_json)
                print(f"Sent message: {message_json}")

                async def send_heartbeat():
                    while signal_handler.can_run():
                        await asyncio.sleep(10)  
                        try:

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
                            
                            if 'threshold' in data and data.get('threshold') is not None:
                                threshold = float(data.get('threshold'))
                            else:    
                                threshold = float(os.getenv("THRESHOLD", 8.0))
                            print(f"Received message: {message_received}")
                            if 'camera_running_status' in data and data['camera_running_status']:
                                    
                                print('going to start_capture_async')
                                set_config_file(data.get('camera_url'), data.get('camera_id'),data.get('camera_type'),data.get('camera_running_status'), threshold)
                                response = await start_capture_async(data.get('camera_url'), data.get('camera_id'), data.get('camera_type'), threshold)
                                response_json = json.dumps(response)
                                await websocket.send(response_json)
                            elif 'camera_running_status' in data and not data['camera_running_status']:

                                set_config_file(data.get('camera_url'), data.get('camera_id'),data.get('camera_type'),data.get('camera_running_status'), threshold)
                                response = await stop_capture_async(data.get('camera_id'))
                                response_json = json.dumps(response)
                                await websocket.send(response_json)
                            elif 'camera_check_url' in data:
                                print('camera_check_url ',data.get('camera_check_url'))

                                response = await cam_check_async(data.get('camera_check_url'), data.get('camera_type'))
                                print('response ',response)
                                response_json = json.dumps(response)
                                await websocket.send(response_json)

                        except:
                            print("Error processing received message.")
                            continue

                        pass

                except websockets.ConnectionClosed:
                    if os.path.exists('camera_config.json'):
                        with open('camera_config.json', 'r') as json_file:
                            data = json.load(json_file)
                            for entry in data:                        
                                camera_id = entry['camera_id']

                                await stop_capture_async(entry['camera_id'])
                                print(f"Camera {entry['camera_id']} has been closed")

                    print("Connection closed by server.")

                finally:

                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            print(f"Connection error: {e}")
            
            print("Reconnecting in 10 seconds...")
            await asyncio.sleep(10)

signal_handler = SignalHandler()

loop = asyncio.get_event_loop()
loop.run_until_complete(connect_to_websocket())