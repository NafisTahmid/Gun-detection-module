import os
import sys
import importlib.util
if getattr(sys, 'frozen', False):
    # When running from a bundled executable
    bundle_dir = sys._MEIPASS
    cv2_path = os.path.join(bundle_dir, "cv2")
    spec = importlib.util.spec_from_file_location("cv2", os.path.join(cv2_path, "cv2.cpython-36m-aarch64-linux-gnu.so"))
    cv2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv2)
    print('is frozen in helper: ',cv2)
    # Check if OpenCV was built with CUDA support
    if cv2.getBuildInformation().find('CUDA') != -1:
        print("OpenCV is built with CUDA support.")
    else:
        print("OpenCV is NOT built with CUDA support.")

else:
    import cv2
# import cv2
import asyncio
import base64
import json
import numpy as np
import websockets
import signal
import sys
import time
import requests
import threading
import queue as thread_queue
import struct
import os
import logging
import string
import random
import subprocess


from dotenv import load_dotenv, set_key
load_dotenv()

user_home = os.getenv("HOME_URL", "/home/root")
app_directory = os.path.join(user_home, 'acceleye-detection-app')
pwd = os.path.dirname(os.path.abspath(__file__))
config_location = os.path.join(app_directory, "server_config.json")

base_uri = os.getenv("URI","https://api.accelx.net/gd_apidev/")
cam_check_url_save = base_uri + "camera/checkcam-url-images/"
capture_interval_seconds = 1


def generate_random_text(length):
    characters = string.ascii_letters + string.digits
    random_text = ''.join(random.choice(characters) for _ in range(length))
    return random_text

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
            print('rtsp')
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



def set_camera_config(camera_url, camera_id, camera_type, camera_running_status, threshold, filename=config_location):
    # Ensure the file exists and has the right structure
    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = {"cameras": []}
    
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