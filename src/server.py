import cv2
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
import helper
from datetime import datetime

import aiohttp
from aiohttp import web
import aiohttp_jinja2
import jinja2

from signal_handler import SignalHandler
import subprocess
import re
from dotenv import load_dotenv, set_key
load_dotenv()

signal_handler = SignalHandler()

# AIOHTTP SERVER CONFIG
HOST = "0.0.0.0"
PORT = 80

fixed_vs_server = os.getenv("SERVER_ID", "acceleye-005a")
secret_key = os.getenv("SECRET_KEY")
base_uri = os.getenv("URI", "https://api.accelx.net/gd_apidev/")
socket_uri = base_uri.replace("http://", "ws://").replace("https://", "wss://")
ws_fixed_url = socket_uri + "ws/fixed-server/" + fixed_vs_server + "/"
cam_check_url_save = base_uri + "camera/checkcam-url-images/"

pwd = os.path.dirname(os.path.abspath(__file__))
config_location = os.path.join(pwd, "server_config.json")
static_path = os.path.join(pwd, 'static')
env_location = os.path.join(pwd, ".env")
# Configuration Constants
frame_no_change_interval = 20
MAX_QUEUE_SIZE = 10
RECONNECT_INTERVAL = 5
capture_interval_seconds = 1
CHUNK_SIZE = 512 * 1024  # 512 KB
ws_url = None

# Global Variables
camera_processes = {}
ws_flag = False
ws = None
ws_fixed = None
video_server_id = None

ws_task = None

def get_ip_addresses(interface):
    try:
        # Run the 'ip addr show' command to get information about the network interface
        output = subprocess.check_output(['ip', 'addr', 'show', interface], universal_newlines=True)
        
        # Find all the 'inet' entries in the output
        ip_addresses = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', output)
        
        return ip_addresses
    except subprocess.CalledProcessError as e:
        return []  # Return an empty list if there's an error
    except Exception as e:
        return []  # Return an empty list if there's an unexpected error

def get_all_interfaces():
    try:
        # Run the 'ip link show' command to get a list of network interfaces
        output = subprocess.check_output(['ip', 'link', 'show'], universal_newlines=True)
        
        # Extract interface names from the output
        interfaces = re.findall(r'^\d+: (\w+):', output, re.MULTILINE)
        
        return interfaces
    except subprocess.CalledProcessError as e:
        return []  # Return an empty list if there's an error
    except Exception as e:
        return []  # Return an empty list if there's an unexpected error

def get_secondary_ip_address():
    try:
        # Get a list of all network interfaces
        interfaces = get_all_interfaces()
        
        # Collect IP addresses from all interfaces
        all_ip_addresses = []
        for interface in interfaces:
            ip_addresses = get_ip_addresses(interface)
            all_ip_addresses.extend(ip_addresses)
        
        # Return the second IP address if available
        if len(all_ip_addresses) >= 2:
            return all_ip_addresses[1]  # Return the second IP address (secondary)
        elif len(all_ip_addresses) == 1:
            return all_ip_addresses[0]  # Return the primary IP address if secondary not found
        else:
            return "No IP address found"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage
ip_address = get_secondary_ip_address()
print(f"Local IP address: {ip_address}")

server_running_status = "No Server Running"
flag = True 

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

    def compare_frame(self, cam_id, target_frame, threshold=8.0):
        if cam_id not in self.reference_frame or self.reference_frame[cam_id].shape != target_frame.shape:
            height, width, channels = target_frame.shape
            self.reference_frame[cam_id] = np.zeros((height, width, channels), dtype=np.uint8)

        reference_gray = cv2.cvtColor(self.reference_frame[cam_id], cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(reference_gray, target_gray)
        mean_diff = frame_diff.mean()
        print(f'mean_diff: {mean_diff} cam_id: {cam_id}')

        if cam_id not in self.last_output_time:
            self.last_output_time[cam_id] = time.time()

        current_time = time.time()
        elapsed_time = current_time - self.last_output_time[cam_id]
        print(f'elapsed_time: {elapsed_time} cam_id: {cam_id}')

        if mean_diff > threshold or elapsed_time >= frame_no_change_interval:
            self.last_output_time[cam_id] = current_time
            self.reference_frame[cam_id] = target_frame
            return True

        return False


def capture_frame(camera_id, cam_type, camera_url, abs_diff_threshold, capture_queue):
    """Thread to capture frames from the camera."""
    latency = 100
    width = 1080
    height = 720

    try:
        if cam_type == 'jpeg':
            while camera_processes.get(camera_id, False):
                try:
                    response = requests.get(camera_url)
                    if response.status_code == 200:
                        frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

                        absdiff_check = Frame.get_instance()
                        accurate = absdiff_check.compare_frame(cam_id=camera_id, target_frame=frame, threshold=abs_diff_threshold)
                        if accurate:
                            frame_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                            frame = {'detect_event': 'Signal','frame':frame_data}
                            capture_queue.put(frame)
                            print(f"Frame for camera {camera_id} added to queue.")
                    else:
                        print(f"Failed to fetch image from URL {camera_url}. Status code: {response.status_code}")
                        no_signal_msg = {'detect_event': 'No Signal'}
                        capture_queue.put(no_signal_msg)
                        time.sleep(5)
                except Exception as e:
                    print(f"Error capturing frame for {camera_id}: {e}")
                    time.sleep(5)
        elif cam_type == 'rtsp':
            gst_str = (
                'rtspsrc async-handling=true location={} latency={} retry=5 ! '
                'rtph264depay ! h264parse ! '
                'queue max-size-buffers=100 leaky=2 ! '
                'nvv4l2decoder enable-max-performance=1 ! '
                'video/x-raw(memory:NVMM), format=(string)NV12 ! '
                'nvvidconv ! video/x-raw, width={}, height={}, format=(string)BGRx ! '
                'videorate ! video/x-raw, framerate=(fraction)1/{} ! '
                'videoconvert ! '
                'appsink'
            ).format(camera_url, latency, width, height, capture_interval_seconds)

            cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

            if not cap.isOpened():
                print(f"Failed to open RTSP stream for camera {camera_id}")
                no_signal_msg = {'detect_event': 'No Signal'}
                capture_queue.put(no_signal_msg)
                return

            while camera_processes.get(camera_id, False):
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to grab frame from camera {camera_id}, retrying...")

                    no_signal_msg = {'detect_event': 'No Signal'}
                    capture_queue.put(no_signal_msg)
                    cap.release()
                    
                    time.sleep(5)
                    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
                    continue

                absdiff_check = Frame.get_instance()
                accurate = absdiff_check.compare_frame(cam_id=camera_id, target_frame=frame, threshold=abs_diff_threshold)

                if accurate:
                    frame_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                    frame = {'detect_event': 'Signal','frame':frame_data}
                    capture_queue.put(frame)
                    print(f"Frame for camera {camera_id} added to queue.")

            cap.release()
    except Exception as e:
        print(f"Error in capture thread for camera {camera_id}: {e}")
    finally:
        print(f"Capture thread for camera {camera_id} has stopped.")

async def send_frame_over_websocket(combined_message):
    """Asynchronous function to send frame data over WebSocket."""
    global ws_flag, ws
    if ws is not None and ws.open:
        try:
            await ws.send(combined_message)
            print("Sent frame.")
        except websockets.ConnectionClosed as e:
            print(f"WebSocket connection closed during send: {e}")
            ws_flag = False
    else:
        print("WebSocket is not connected.")


async def send_frame_over_no_signal_websocket(combined_message):
    """Asynchronous function to send frame data over WebSocket."""
    global ws_flag, ws
    if ws is not None and ws.open:
        try:
            await ws.send(json.dumps(combined_message))
            print("Sent frame.")
        except websockets.ConnectionClosed as e:
            print(f"WebSocket connection closed during send: {e}")
            ws_flag = False
    else:
        print("WebSocket is not connected.")

def send_frames(camera_id, capture_queue, loop):
    """Thread to send frames over WebSocket."""
    global ws_flag, ws, signal_handler
    while signal_handler.can_run():
        try:
            frame_data = capture_queue.get()
            frame_ch = frame_data['detect_event']

            if 'detect_event' in frame_data and frame_data['detect_event'] == 'No Signal':

                no_signal_msg = {'cam_id': camera_id, 'detect_event': 'No Signal'}
                future = asyncio.run_coroutine_threadsafe(send_frame_over_no_signal_websocket(no_signal_msg), loop)
                future.result()  # Wait for send to complete or raise an exception

                time.sleep(0.1)
            
            else:
                camera_id_bytes = f"{camera_id}|".encode('utf-8')
                frame = frame_data['frame']
                combined_message = camera_id_bytes + frame

                # Use asyncio.run_coroutine_threadsafe to run the coroutine in the main event loop
                future = asyncio.run_coroutine_threadsafe(send_frame_over_websocket(combined_message), loop)
                future.result()  # Wait for send to complete or raise an exception

            time.sleep(0.1)
        except websockets.ConnectionClosed as e:
            print(f"WebSocket connection closed: {e}. Reconnecting...")
            ws_flag = False
            time.sleep(RECONNECT_INTERVAL)
            # Reconnect logic is handled by the main loop
        except Exception as e:
            print(f"Error while sending frame for camera {camera_id}: {e}")
            time.sleep(2)

async def websocket_manager(app):
    """Manages the lifecycle of the WebSocket connection."""
    loop = asyncio.get_event_loop()
    
    # Task for WebSocket connection
    ws_task = loop.create_task(connect_ws_fixed(loop))

    # Wait for the app to cleanup (shutting down)
    yield

    # When shutting down, cancel the WebSocket task
    print("Shutting down WebSocket connection...")
    ws_task.cancel()
    try:
        await ws_task
    except asyncio.CancelledError:
        print("WebSocket task cancelled.")
# Capture and WebSocket logic remains unchanged (send_frame_over_websocket, capture_frame, etc.)

async def connect_ws_fixed(loop):
    """Connect to the fixed WebSocket and listen for the secret_key."""
    global ws_fixed_url, ws_fixed, ws_url, fixed_vs_server, signal_handler, video_server_id
    while signal_handler.can_run():
        try:
            print("Connecting to fixed WebSocket...")
            ws_fixed = await websockets.connect(ws_fixed_url)
            print("Connected to fixed WebSocket.")
            while signal_handler.can_run():
                subs_url = base_uri + 'service/service-filter-vs/'
                msg_send = {"server_fixed_id": fixed_vs_server, "ip_address": ip_address}
                print(f'msg_send: {msg_send}')
                await ws_fixed.send(json.dumps(msg_send))

                message = await ws_fixed.recv()
                data = json.loads(message)
                print(f"Received message on fixed WebSocket: {data}")
                
                # Check for secret_key in the received data
                if 'server_skey' in data:
                    server_skey = data.get('server_skey')
                    video_server_id = data.get('server_new_id')
                    print(f"Received secret_key: {server_skey}. Now connecting to the main WebSocket...")
                    ws_url = socket_uri + 'ws/video-process/' + video_server_id + '/'

                    print(f'new ws url: {ws_url}')
                    
                    # After receiving the secret_key, connect to the main WebSocket (ws_url)
                    await reconnect_websocket(loop)
                    await websocket_listener(loop)
                else:
                    await asyncio.sleep(10)
        except websockets.ConnectionClosed as e:
            print(f"Fixed WebSocket connection closed: {e}. Reconnecting...")
            await asyncio.sleep(RECONNECT_INTERVAL)
        except Exception as e:
            print(f"Error in fixed WebSocket: {e}")
            await asyncio.sleep(1)


async def reconnect_websocket(loop):
    """Reconnection logic for WebSocket."""
    global ws_flag, ws, ws_url, fixed_vs_server, video_server_id
    while not ws_flag:
        try:
            ws = await websockets.connect(ws_url, ping_interval=10)
            send_status = {'fixed_vs_server': fixed_vs_server, 'video_server_id': video_server_id, 'msg': 'video server for NON ML SERVER'}
            await ws.send(json.dumps(send_status))
            ws_flag = True
            print("WebSocket connected")
        except Exception as e:
            print(f"Failed to reconnect WebSocket: {e}")
            await asyncio.sleep(RECONNECT_INTERVAL)


async def websocket_listener(loop):
    """Asynchronous listener for WebSocket messages."""
    global ws, ws_flag,video_server_id, signal_handler, server_running_status
    first_time_call = True
    while signal_handler.can_run():
        try:
            server_running_status = f"Server Is Running: {video_server_id}"
            message_to_send = {"message": f"message from {video_server_id} server", "status_code": 2000, "video_process_server_id": video_server_id, "ip_address": ip_address}     
            await ws.send(json.dumps(message_to_send))

            if ws_flag:
                server_running_status = f"Server Is Running: {video_server_id}"
                if os.path.exists(config_location) and first_time_call:
                    with open(config_location, 'r') as json_file:
                        cam_data = json.load(json_file)
                    cameras = cam_data.get('cameras', [])
                    for entry in cameras:
                        camera_id = entry['camera_id']
                        camera_url = entry['camera_url']
                        camera_type = entry['camera_type']
                        camera_running_status = entry['camera_running_status']
                        threshold = entry['threshold']
                        #if camera_running_status == 'true': for browser websoket testing
                        if camera_running_status == True:

                            capture_queue = thread_queue.Queue(maxsize=MAX_QUEUE_SIZE)
                            camera_processes[camera_id] = True
                            print(f'camera_id; {camera_id}')
                            print('camera process: ',camera_processes[camera_id])

                            # Start capture and send threads
                            capture_thread = threading.Thread(
                                target=capture_frame, 
                                args=(camera_id, camera_type, camera_url, threshold, capture_queue)
                            )
                            capture_thread.daemon = True
                            capture_thread.start()

                            send_thread = threading.Thread(
                                target=send_frames, 
                                args=(camera_id, capture_queue, loop)
                            )
                            send_thread.daemon = True
                            send_thread.start()

                            print(f"Started capturing for camera {camera_id}. From Config List")
                            cam_run_status = {"camera_id":camera_id,"status_code": 200,"msg":"Camera running from video server."}

                            await ws.send(json.dumps(cam_run_status))

                    
                    first_time_call = False
                
                message = await ws.recv()
                server_running_status = f"Server Is Running: {video_server_id}"
                data = json.loads(message)
                print(f"Received data: {data}")

                if 'camera_id' in data and 'camera_running_status' in data:
                    camera_id = data.get('camera_id')
                    camera_run = data.get('camera_running_status', False)
                    #if camera_run.lower() == 'true': for sending browser socket for test
                    if camera_run == True:
                        if camera_id in camera_processes and camera_processes[camera_id]:
                            print(f"Camera {camera_id} is already running.")
                            run_status = {"camera_id":camera_id,"status_code": 2004,"msg":"Camera already running."}
                            await ws.send(json.dumps(run_status))
                        else:
                            camera_url = data.get('camera_url')
                            cam_type = data.get('camera_type')
                            abs_diff_threshold = float(data.get('threshold', 5.0))
                            helper.set_camera_config(camera_url, data.get('camera_id'), cam_type, data.get('camera_running_status'),abs_diff_threshold)
                            capture_queue = thread_queue.Queue(maxsize=MAX_QUEUE_SIZE)
                            camera_processes[camera_id] = True

                            # Start capture and send threads
                            capture_thread = threading.Thread(
                                target=capture_frame, 
                                args=(camera_id, cam_type, camera_url, abs_diff_threshold, capture_queue)
                            )
                            capture_thread.daemon = True
                            capture_thread.start()

                            send_thread = threading.Thread(
                                target=send_frames, 
                                args=(camera_id, capture_queue, loop)
                            )
                            send_thread.daemon = True
                            send_thread.start()

                            print(f"Started capturing for camera {camera_id}.")
                    else:
                        
                        if camera_id not in camera_processes or not camera_processes[camera_id]:
                            #print('camera_processes: ',camera_processes[camera_id])
                            #print('cam process ',camera_processes)
                            print(f"Camera {camera_id} is already stopped.")
                            stop_status = {"camera_id":camera_id,"status_code": 2004,"msg":"Camera already stopped."}
                            await ws.send(json.dumps(stop_status))
                        else:
                            camera_processes[camera_id] = False
                            print(f"Stopped capturing for camera {camera_id}.")
                            abs_diff_threshold = float(data.get('threshold', 5.0))

                            helper.set_camera_config(data.get('camera_url'), data.get('camera_id'), data.get('camera_type'), data.get('camera_running_status'), abs_diff_threshold)
                elif 'camera_check_url' in data:
                    print('camera_check_url ', data.get('camera_check_url'))
                    
                    response = await helper.cam_check_async(data.get('camera_check_url'), data.get('camera_type'),video_server_id)
                    
                    response_json = json.dumps(response)
                    print(f'response_json: {response_json}')
                    await ws.send(response_json)
                
                elif 'server_restart_trigger' in data:
                    print('restarting server')
                    await helper.restart_service()
        except websockets.ConnectionClosed as e:
            print(f"WebSocket connection closed: {e}. Reconnecting...")
            ws_flag = False
            await reconnect_websocket(loop)
        except Exception as e:
            print(f"Error in websocket_listener: {e}")
            await asyncio.sleep(1)

def remove_config_file():
    """Removes the configuration file."""
    config_location = os.path.join(pwd, "server_config.json")
    if os.path.exists(config_location):
        os.remove(config_location)
        print(f"Configuration file '{config_location}' removed.")

async def restart_websocket_connection(loop, new_video_server_id):
    """Restarts the WebSocket connection."""
    global ws_task, video_server_id

    print(f"Restarting WebSocket connection for server ID: {video_server_id}...")

    # Cancel the previous WebSocket task if it exists
    if ws_task is not None:
        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            print("Previous WebSocket task cancelled.")

    # Remove the config file before starting the new connection
    remove_config_file()
    video_server_id = new_video_server_id

    # Start a new WebSocket connection (run in the background)
    #ws_task = loop.create_task(connect_ws_fixed(loop,video_server_id))
    await helper.reset_server()
    await helper.restart_service()

# Background task for updating env and restarting WebSocket
async def update_server_id_and_restart_ws(video_server_id):
    # Update the .env file
    set_key(env_location, 'SERVER_ID', video_server_id)

    # Get the current event loop
    loop = asyncio.get_event_loop()

    # Restart the WebSocket connection
    await restart_websocket_connection(loop)


async def restart_server(request):
    try:
        request_data = await request.json()
        trigger = request_data.get('trigger')
        
        if trigger == 1:
            await helper.reset_server()
        
        
        await helper.restart_service()
        return web.json_response({'status': 'success', 'message': 'Server restarted successfully'})
    except Exception as e:
        return web.json_response({'status': 'error', 'message': str(e)})

async def check_server(request):
    try:
        data = server_running_status
        return web.json_response({'status': 'success', 'message': str(data)})
    except Exception as e:
        return web.json_response({'status': 'error', 'message': str(e)})

async def server_config(request):
    try:
        # Open and read the JSON file
        with open(config_location, 'r') as file:
            data = json.load(file)
        
        # Return JSON response
        return web.json_response(data)
    except Exception as e:
        # Handle errors (e.g., file not found, JSON decode error)
        return web.json_response({'error': str(e)}, status=500)
    

@aiohttp_jinja2.template('index.html')
async def index(request):
    global server_running_status, flag, fixed_vs_server, ip_address

    if fixed_vs_server and flag:
        ws_url = fixed_vs_server
        #asyncio.ensure_future(restart_websocket_connection(ws_url, fixed_server=True))

    context = {
        'fixed_vs_server': fixed_vs_server,
        'ip_address': ip_address,
        'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'video_server_status': server_running_status
    }

    if request.method == 'POST':
        data = await request.post()
        fx_video_server_id = data.get('video_server_id')

        if fx_video_server_id:
            loop = asyncio.get_event_loop()
            loop.create_task(update_server_id_and_restart_ws(fx_video_server_id))
        
            print('new server')
            context['video_server_status'] = f'Video server {video_server_id} is running'
            return web.json_response({'video_server_status': context['video_server_status']})
        else:
            return web.json_response({'video_server_status': 'Please enter valid details'})

    return context


async def on_shutdown(app):
    """Gracefully shut down the WebSocket and frame processes when the server is stopped."""
    print("Shutting down all camera processes...")

    for camera_id in camera_processes:
        camera_processes[camera_id] = False


async def init_app(loop):
    # Create an aiohttp app and set up routes
    app = web.Application()
    aiohttp_jinja2.setup(
        app, loader=jinja2.FileSystemLoader(os.path.join(pwd, 'templates'))
    )
    app.router.add_get('/', index)
    
    app.router.add_post('/', index)
    app.router.add_static('/static', static_path)
    app.router.add_post('/restart_server', restart_server)
    app.router.add_get('/check_server', check_server)
    app.router.add_get('/server_config', server_config)

    app.cleanup_ctx.append(websocket_manager)

    return app


def shutdown(signal_received, frame):
    """Handle shutdown signals."""
    print("Shutting down all camera processes...")

    for camera_id in camera_processes:
        camera_processes[camera_id] = False
    sys.exit(0)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    
    # Serve aiohttp webserver
    app = loop.run_until_complete(init_app(loop))
    web.run_app(app, host=HOST, port=PORT)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        shutdown(None, None)
    finally:
        loop.close()
