import uuid
import websockets
import time
import json 
import os
import asyncio
import string
import random
import cv2
import threading
import queue
import requests
import numpy as np
import logging
from dotenv import load_dotenv
import subprocess
from signal_handler import SignalHandler


# Thread Settings
camera_threads = {}
frame_queues = {}
max_queue_size = 30
max_processing_threads = 3
index = 0

# .env Settings
load_dotenv()
fixed_vs_server = os.getenv("SERVER_ID")
group_name = os.getenv("SERVER_ID")
secret_key = os.getenv("SECRET_KEY")
base_uri = os.getenv("URI")
if base_uri.startswith("http://"):
    socket_uri = base_uri.replace("http://", "ws://")
else:
    socket_uri = base_uri.replace("https://", "wss://")

connection_uri = socket_uri + "ws/video-process/" + group_name + "/"
detect_image_save = base_uri + "camera/weapon-detect-images-create/"
# detect_image_save = base_uri + "camera/weapon-detect-images/"
camera_setups_url = base_uri + "camera/camera-setups/"
cam_check_url_save = base_uri + "camera/checkcam-url-images/"
headers = {
            'accept': 'application/json',
            'X-Custom-Secret-Key': secret_key,
            'X-Custom-Server-Fixed-Id': fixed_vs_server
        }


capture_interval_seconds = int(os.getenv("CAPTURE_INTERVAL_SECONDS", 3))
frame_no_change_interval = int(os.getenv("FRAME_NO_CHANGE_INTERVAL", 20))
weapon_detection_url = os.getenv("WEAPON_DETECTION_URL", "http://192.168.1.52:8080/predictions/accelx-weapon-dt-yolos-detr")

ml_server = os.getenv("SERVICE_URL", "http://192.168.1.52:8080/predictions/accelx-weapon-dt-yolos-detr")
# Log
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
pwd = os.path.dirname(os.path.abspath(__file__))
config_location = os.path.join(pwd, "server_config.json")
# Initialize the SignalHandler
signal_handler = SignalHandler()


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

async def restart_service():
    try:
        # Use asyncio.create_subprocess_exec for non-blocking subprocess call
        process = await asyncio.create_subprocess_exec(
            'sudo', 'systemctl', 'restart', 'videoprocess',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait until process completes
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print("Service 'videoprocess' has been restarted successfully.")
        else:
            error_message = stderr.decode().strip()
            print(f"Failed to restart service 'videoprocess': {error_message}")
            raise subprocess.CalledProcessError(process.returncode, 'systemctl restart videoprocess')
    except asyncio.CancelledError:
        print("Service restart was cancelled.")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}: {e.cmd}")
    except Exception as e:
        print(f"Exception occurred: {e}")
        raise

async def reset_server():
    command = ['sudo', 'rm', config_location]
    
    try:
        # Create a subprocess to run the command asynchronously
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for the command to complete
        stdout, stderr = await process.communicate()

        # Check if the command was successful
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with error: {stderr.decode()}")

        return "Server configuration reset successfully."

    except Exception as e:
        return f"An error occurred: {str(e)}"

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
    count = {}
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
                    response = requests.post(detect_image_save, files=files, headers=headers, data=data)
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
        n[cam_id] = 0
        count[cam_id] = 0
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
                response = requests.post(detect_image_save, files=files, headers=headers, data=data)
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
                    print(f"Cam ID: {cam_id}! Ready For Processing")
                    frame_queue.put(frame)
                    continue
                
            

        if cap is not None:
            print("cap is released")
            print(camera_threads[cam_id],camera_threads[cam_id].capture_thread.stopped.is_set())
            cap.release()

def process_frames(thread_id, cam_id):
    global frame_queues, max_processing_threads, signal_handler
    if os.path.exists(config_location):
        with open(config_location, 'r') as json_file:
            subs_list = json.load(json_file)
    else:
        subs_list = {"subscriptions":[]}

    while cam_id in camera_threads and not (signal_handler.shutdown_requested or camera_threads[cam_id].capture_thread.stopped.is_set()):

        if cam_id not in frame_queues:
            print(f"Thread {thread_id} for Camera {cam_id}: Frame queue not found.")
            continue

        frame_queue = frame_queues[cam_id]

        try:
            frame = frame_queue.get(timeout=1)
            frame_violence = frame.copy()
            frame_cam = cv2.putText(frame_violence, f"Camera {cam_id}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            processed_frame = cv2.putText(frame, f"Camera {cam_id}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
            # print(f"Cam ID: {cam_id} Starting To Process")
            
            service_api_call = subs_list.get("subscriptions", [])
            services = [(entry["subscription_name"], entry["API_endpoint"]) for entry in service_api_call]
            for name, endpoint in services:
                print(f"Subscription Name: {name}, API Endpoint: {endpoint}")
                name_lower = name.lower()
                if "gun" in name_lower:
                    print(f"Calling Gun API for subscription: {name}")
                    weapon_image = cv2.imencode('.jpg', processed_frame)[1].tobytes()
                    gun_url = ml_server + endpoint
                    
                    res = requests.post(gun_url, data=weapon_image)
                    detect = json.loads(res.content)

                    if 'probs' in detect and 'bboxes_scaled' in detect:
                        bboxes_list = detect['bboxes_scaled']

                        probs_list = detect['probs']
                        weapon = detect['id2label']['0']
                        print(weapon)
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
                            response = requests.post(detect_image_save, files=files, headers=headers, data=data)
                            print(f"Thread {thread_id} for Camera {cam_id}: Frame processed with Event detect.")

                        if not bboxes_list:
                            files = {'detect_image': (f'frame_{cam_id}.jpg', weapon_image, 'image/jpeg')}

                            data = {'camera': cam_id, 'detect_event': 'No Event'}
                            response = requests.post(detect_image_save, files=files, headers=headers, data=data)

                            print(f"Thread {thread_id} for Camera {cam_id}: Frame processed with NO Event detect.")
                    
                    else:
                        processed_frame_d = cv2.putText(frame, f"Status: ML Server Not Working", (10, 70),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        weapon_image = cv2.imencode('.jpg', processed_frame_d)[1].tobytes()
                        random_text = generate_random_text(6)

                        files = {'detect_image': (f'frame_{random_text}.jpg', weapon_image, 'image/jpeg')}
                        print("random_text: ", random_text)

                        data = {'camera': cam_id, 'detect_event': 'Detection N/A'}
                        print("Data:" , data)
                        response = requests.post(detect_image_save, files=files, headers=headers, data=data)
                        print("response:", response)
                        print(f"Requested method is not allowed, please refer to API document for Camera {cam_id}")

                    #.......gun end.......#
                elif "violence" in name_lower:
                    print(f"Calling violence API for subscription: {name}")
                    random_text = generate_random_text(6)
                    violence_image = cv2.imencode('.jpg', frame_cam)[1].tobytes()
                    print('process frame')

                    violence_url = ml_server + endpoint
                    
                    res = requests.post(violence_url, data=violence_image)
                    print(f'process frame: {res.status_code}')
                    detect = json.loads(res.content)
                    print(f'process frame detect: {detect}')
                    
                    if res.status_code == 200:
                        
                        files = {'detect_image': (f'frame_v_{random_text}.jpg', violence_image, 'image/jpeg')}
                        print("if random_text: ", random_text)
                        #  detect['label']
                        data = {'camera': cam_id, 'detect_event': detect['label']}
                        print("vio if Data:" , data)
                        response = requests.post(detect_image_save, files=files, headers=headers, data=data)
                        print("response:", response)
                        #print(f"Requested method is not allowed, please refer to API document for Camera {cam_id}")
                        print(f'detect value for violence: {detect}')
                    else:
                        print(f'process frame detect: {detect}')
                        files = {'detect_image': (f'frame_{random_text}.jpg', violence_image, 'image/jpeg')}
                        print("el random_text: ", random_text)

                        data = {'camera': cam_id, 'detect_event': 'No Violence'}
                        print("vio else Data:" , data)
                        response = requests.post(detect_image_save, files=files, headers=headers, data=data)
                        print("response:", response)
                        #print(f"Requested method is not allowed, please refer to API document for Camera {cam_id}")
                        print(f'detect value for no violence: {detect}')
                
                elif "fire" in name_lower:
                    print(f"Calling Fire API for subscription: {name}")
                    random_text = generate_random_text(6)
                    fire_image = cv2.imencode('.jpg', frame_cam)[1].tobytes()

                    fire_url = ml_server + endpoint
                    
                    res = requests.post(fire_url, data=fire_image)
                    detect = json.loads(res.content)
                    
                    if res.status_code == 200:
                        
                        
                        files = {'detect_image': (f'frame_{random_text}.jpg', fire_image, 'image/jpeg')}
                        print("random_text: ", random_text)

                        data = {'camera': cam_id, 'detect_event': detect}
                        print("Data:" , data)
                        response = requests.post(detect_image_save, files=files, headers=headers, data=data)
                        print("response:", response)
                        #print(f"Requested method is not allowed, please refer to API document for Camera {cam_id}")
                        print(f'detect value for fire: {detect}')
                    else:
                        
                        files = {'detect_image': (f'frame_{random_text}.jpg', fire_image, 'image/jpeg')}
                        print("random_text: ", random_text)

                        data = {'camera': cam_id, 'detect_event': 'No Fire'}
                        print("Data:" , data)
                        response = requests.post(detect_image_save, files=files, headers=headers, data=data)
                        print("response:", response)
                        #print(f"Requested method is not allowed, please refer to API document for Camera {cam_id}")
                        print(f'detect value for no fire: {detect}')

                else:
                    print(f"service name not matching: {name}")

            

        except queue.Empty:

            pass
        except Exception as e:
            processed_frame = cv2.putText(frame_cam, f"Status: ML Server Not Working", (10, 70),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            weapon_image = cv2.imencode('.jpg', processed_frame)[1].tobytes()
            random_text = generate_random_text(6)

            files = {'detect_image': (f'frame_{random_text}.jpg', weapon_image, 'image/jpeg')}
            print("random_text: ", random_text)

            data = {'camera': cam_id, 'detect_event': 'API N/A'}
            print("Data:" , data)
            response = requests.post(detect_image_save, files=files, headers=headers, data=data)
            print("response:", response)
            print(f"Error processing frame: {e} for Camera {cam_id}")

def create_processing_thread(thread_id, cam_id):
    new_thread = threading.Thread(target=process_frames, args=(thread_id, cam_id))
    new_thread.start()
    return new_thread

def set_camera_config(camera_url, camera_id, camera_type, camera_running_status, threshold, filename=config_location):
    # Ensure the file exists and has the right structure
    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = {"cameras": [], "subscriptions": []}
    
    # Check if the camera_id already exists
    cameras = data.get("cameras", [])
    found = False
    for entry in cameras:
        if entry["camera_id"] == camera_id:
            entry["camera_url"] = camera_url
            entry["camera_type"] = camera_type
            entry["camera_running_status"] = camera_running_status
            entry["threshold"] = threshold
            found = True
            break
    
    if not found:
        cameras.append({
            "camera_id": camera_id,
            "camera_url": camera_url,
            "camera_type": camera_type,
            "camera_running_status": camera_running_status,
            "threshold": threshold
        })
    
    # Write the updated data back to the file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def unset_camera_config(camera_id, filename=config_location):
    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        
        # Ensure the structure is present
        if "cameras" in data:
            data["cameras"] = [entry for entry in data["cameras"] if entry["camera_id"] != camera_id]
        
        # Write the updated data back to the file
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    else:
        print("Config file not found.")
        
def set_subscription(subscription_name, API_endpoint, filename=config_location):
    
    # Ensure the file exists and has the right structure
    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = {"cameras": [], "subscriptions": []}
    
    # Check if the subscription_name already exists
    subscriptions = data.get("subscriptions", [])
    found = False
    for entry in subscriptions:
        if entry["subscription_name"] == subscription_name:
            entry["API_endpoint"] = API_endpoint
            found = True
            break
    
    if not found:
        subscriptions.append({
            "subscription_name": subscription_name,
            "API_endpoint": API_endpoint
        })
    
    # Write the updated data back to the file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    

def unset_subscription(subscription_name, filename=config_location):
    try:
        if os.path.exists(filename):
            # Load the existing configuration
            with open(filename, 'r') as json_file:
                data = json.load(json_file)

            # Ensure the 'subscriptions' key exists
            if "subscriptions" in data:
                # Filter out the subscription to be removed
                updated_subscriptions = [entry for entry in data["subscriptions"] if entry["subscription_name"] != subscription_name]

                # Update the subscriptions list
                data["subscriptions"] = updated_subscriptions

                # Write the updated configuration back to the file
                with open(filename, 'w') as json_file:
                    json.dump(data, json_file, indent=4)

                print(f"Subscription '{subscription_name}' removed successfully.")
            else:
                print("No 'subscriptions' key found in the config file.")
        else:
            print("Config file not found.")
    except Exception as e:
        print(f"Error unsetting subscription '{subscription_name}': {str(e)}")

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


async def stop_all_threads():
    global camera_threads, frame_queues

    try:
        if not camera_threads:
            raise ValueError("No captures are running.")

        # Stop and join all capture and processing threads
        for cam_id in list(camera_threads.keys()):
            print(f'stop camera thread for cam_id: {cam_id}')
            camera_threads[cam_id].capture_thread.stop()
            camera_threads[cam_id].capture_thread.join()

            for process_thread in camera_threads[cam_id].process_threads:
                process_thread.join()

            # Remove the camera_thread and frame_queue for this cam_id
            del camera_threads[cam_id]
            del frame_queues[cam_id]

        return {"message": "All captures stopped and resources cleaned up."}
    except Exception as e:
        return {"error": str(e)}

async def cam_check_async(cam_url, cam_type, group_name):
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
                print(f"random_text: {random_text},group_name: {group_name}")

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


async def update_service(subs_url, server_key, server_fixed_id):
    try:
        params = {'server_fixed_id': server_fixed_id}  
        headers = {
            'accept': 'application/json',
            'X-Custom-Secret-Key': server_key
        }
        response = requests.get(subs_url, headers=headers, params=params)

        if response.status_code == 200:
            data_js = response.json()
            print(f'subs: {data_js}')
            for subscription in data_js:
                name = subscription.get('name')
                api_endpoint = subscription.get('api_endpoint')
                
                # Call the set_subscription function
                set_subscription(subscription_name=name, API_endpoint=api_endpoint)
        else:
            print(f"Request failed with status code {response.status_code}")

        return {"message": f"Subscription api call {subs_url} with fixed id {server_fixed_id}"}
    except Exception as e:
        return {"error": str(e)}
    
async def unset_service(subs_url, server_key, server_fixed_id, config_path=config_location):
    try:
        params = {'server_fixed_id': server_fixed_id}
        headers = {
            'accept': 'application/json',
            'X-Custom-Secret-Key': server_key
        }
        response = requests.get(subs_url, headers=headers, params=params)

        if response.status_code == 200:
            data_js = response.json()
            print(f'subs: {data_js}')

            # Load current configuration from config_location
            if os.path.exists(config_path):
                with open(config_path, 'r') as config_file:
                    server_config = json.load(config_file)
            else:
                server_config = {}

            # Get the list of current subscription names from the API response
            current_subs = {sub['name'] for sub in data_js}

            # Iterate through the subscriptions in the config file
            updated_subs = []
            for sub in server_config.get('subscriptions', []):
                if sub['subscription_name'] in current_subs:
                    updated_subs.append(sub)
                else:
                    print(f"Removing subscription: {sub['subscription_name']}")
                    unset_subscription(subscription_name=sub['subscription_name'])

            # Save the updated configuration back to config_location
            server_config['subscriptions'] = updated_subs
            with open(config_path, 'w') as config_file:
                json.dump(server_config, config_file, indent=4)

        else:
            print(f"Request failed with status code {response.status_code}")

        return {"message": f"Subscription unset call {subs_url} with fixed id {server_fixed_id}"}
    except Exception as e:
        return {"error": str(e)}