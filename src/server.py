# server.py
import aiohttp
import signal
from aiohttp import web
import aiohttp_jinja2
import jinja2
import asyncio
import socket
import websockets
from datetime import datetime
import os
import json
import requests
import time
from dotenv import load_dotenv, set_key
import subprocess
import re

# Import the SignalHandler class
from signal_handler import SignalHandler

# Load environment variables
load_dotenv()

import utils

# Thread Settings
camera_threads = {}
frame_queues = {}
max_queue_size = 30
max_processing_threads = 3

# .env Settings
fixed_vs_server = os.getenv("SERVER_ID")
secret_key = os.getenv("SECRET_KEY")
base_uri = os.getenv("URI")
socket_uri = base_uri.replace("http://", "ws://").replace("https://", "wss://")

detect_image_save = base_uri + "camera/weapon-detect-images/"
camera_setups_url = base_uri + "camera/camera-setups/"
cam_check_url_save = base_uri + "camera/checkcam-url-images/"

ins_id_url = base_uri + "user/videoprocess-list/?video_process_server_id=" + fixed_vs_server
print(f'video server list_url: {ins_id_url}')

capture_interval_seconds = int(os.getenv("CAPTURE_INTERVAL_SECONDS", 3))
frame_no_change_interval = int(os.getenv("FRAME_NO_CHANGE_INTERVAL", 20))
weapon_detection_url = os.getenv("WEAPON_DETECTION_URL", "http://192.168.1.52:8080/predictions/accelx-weapon-dt-yolos-detr")

pwd = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(pwd, 'static')
template_dir = os.path.join(pwd, 'templates')
config_location = os.path.join(pwd, "server_config.json")
env_location = os.path.join(pwd, ".env")
# Global variables

def get_secondary_ip_address(interface='eth0'):
    try:
        # Run the 'ip addr show' command to get information about the network interface
        output = subprocess.check_output(['ip', 'addr', 'show', interface], universal_newlines=True)
        
        # Find all the 'inet' entries in the output
        ip_addresses = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', output)
        
        if len(ip_addresses) >= 2:
            return ip_addresses[1]  # Return the second IP address (secondary)
        elif len(ip_addresses) == 1:
            return ip_addresses[0]  # Return the primary IP address if secondary not found
        else:
            return "No IP address found"
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage
interface = 'eth0'  # Replace with your network interface name
ip_address = get_secondary_ip_address(interface)
print(f"Local IP address: {ip_address}")

fixed_server = True

# Set up aiohttp app and Jinja2 template rendering
app = web.Application()
aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(template_dir))

# Initialize the SignalHandler


signal_handler = SignalHandler()
server_running_status = "No Server Running"
flag = True 



async def restart_server(request):
    try:
        request_data = await request.json()
        trigger = request_data.get('trigger')
        
        if trigger == 1:
            await utils.reset_server()
        
        
        await utils.restart_service()
        return web.json_response({'status': 'success', 'message': 'Server restarted successfully'})
    except Exception as e:
        return web.json_response({'status': 'error', 'message': str(e)})

# Add the new endpoint to your application

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
    

app.router.add_post('/restart_server', restart_server)
app.router.add_get('/check_server', check_server)
app.router.add_get('/server_config', server_config)

@aiohttp_jinja2.template('index.html')
async def index(request):
    global server_running_status
    global flag
    if fixed_vs_server and flag:
        ws_url = fixed_vs_server
        #await restart_websocket_connection(ws_url, fixed_server=True)
        asyncio.ensure_future(restart_websocket_connection(ws_url, fixed_server=True))

    context = {
        'fixed_vs_server': fixed_vs_server,
        'ip_address': ip_address,
        'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'video_server_status': server_running_status
    }

    if request.method == 'POST':
        data = await request.post()
        video_server_id = data.get('video_server_id')

        if video_server_id:
            set_key(env_location, 'SERVER_ID', video_server_id)
            asyncio.ensure_future(restart_websocket_connection(video_server_id, fixed_server=True))
            context['video_server_status'] = f'Video server {video_server_id} is running'
            return web.json_response({'video_server_status': context['video_server_status']})
        else:
            return web.json_response({'video_server_status': 'Please enter valid details'})

    return context

# app.router.add_post('/', index)
# app.router.add_static('/static', 'static')

async def restart_websocket_connection(ws_url, fixed_server=True):
    global fixed_vs_server
    print(f'ws_url in restart: {ws_url} and fixed_server: {fixed_server}')

    
    
    if 'websocket_task' in app and not app['websocket_task'].done():
        app['websocket_task'].cancel()
        try:
            await app['websocket_task']
        except asyncio.CancelledError:
            print("Previous WebSocket task cancelled successfully.")

    app['websocket_task'] = asyncio.ensure_future(connect_to_websocket(ws_url, fixed_server))

async def connect_to_websocket(ws_url, fixed_server):
    global server_running_status, signal_handler
    global fixed_vs_server
    global flag

    uri = socket_uri + ("ws/fixed-server/" if fixed_server else "ws/video-process/") + ws_url + "/"
    print(f"socket url: {uri}")

    while signal_handler.can_run():
        try:
            async with websockets.connect(uri) as websocket:
                print(f"Connected to WebSocket server: fixed_server - {fixed_server}")
                flag = False 
                await asyncio.sleep(2)
                server_running_status = f"Server Is Running: {ws_url}"
                print(f'server running status - {server_running_status}')

                if fixed_server:

                    subs_url = base_uri + 'service/service-filter-vs/'

                    msg_send = {"server_fixed_id": ws_url, "ip_address": ip_address}
                    print(f'msg_send: {msg_send}')
                    msg_send_json = json.dumps(msg_send)
                    await websocket.send(msg_send_json)
                    msg = await websocket.recv()
                    msg_json = json.loads(msg)
                    print(f'msg_json: {msg_json}')
                    if 'status_code' in msg_json:
                        if msg_json.get('status_code') == 2000:
                            server_skey = msg_json.get('server_skey')
                            server_new_id = msg_json.get('server_new_id')
                            set_key(env_location, 'SECRET_KEY', server_skey)
                            fixed_server = False
                            fixed_vs_server = ws_url
                            print(f'server_new_id: {server_new_id} and server_skey: {server_skey} and fixed server: {fixed_server}')
                            print('#######....done restart.....##########')
                            
                            subs_call = await utils.update_service(subs_url,server_skey,fixed_vs_server)
                            subs_call_2 = await utils.unset_service(subs_url,server_skey,fixed_vs_server)
                            print('subs_call ', subs_call)
                            print('subs_call_2', subs_call_2)

                            await restart_websocket_connection(server_new_id, fixed_server=False)
                            break
                        else:
                            server_skey = msg_json.get('server_skey')
                            set_key(env_location, 'SECRET_KEY', server_skey)
                            fixed_server = True
                            fixed_vs_server = ws_url
                            print(f'server_skey: {server_skey} and fixed server: {fixed_server}')
                            print('#######....done restart.....##########')

                            subs_call = await utils.update_service(subs_url,server_skey,fixed_vs_server)
                            subs_call_2 = await utils.unset_service(subs_url,server_skey,fixed_vs_server)
                            print('subs_call ', subs_call)
                            print('subs_call_2', subs_call_2)

                            await restart_websocket_connection(ws_url, fixed_server=True)
                            break

                else:
                    if os.path.exists(config_location):
                        with open(config_location, 'r') as json_file:
                            data = json.load(json_file)
                            cameras = data.get('cameras', [])
                        for entry in cameras:
                            camera_id = entry['camera_id']
                            api_url = f'{camera_setups_url}{camera_id}/'
                            cam_found = requests.get(api_url)
                            cam_info = cam_found.json()

                            if cam_found.status_code == 200:
                                video_process_server_id = cam_info['video_process_server_info']['video_process_server_id']
                                if video_process_server_id == ws_url:
                                    if entry['camera_running_status'] == True:
                                        response = await utils.start_capture_async(entry['camera_url'], entry['camera_id'], entry['camera_type'], entry['threshold'])
                                        response_json = json.dumps(response)
                                        await websocket.send(response_json)
                                        patch_data = {"camera_running_status": True, "camera_frame_cap_status": True}
                                        print(api_url)
                                        resf = requests.patch(api_url, json=patch_data)
                                        if resf.status_code == 200:
                                            print("Success")
                                        else:
                                            await utils.stop_capture_async(entry['camera_id'])
                                            print(f"Camera {entry['camera_id']} didn't start")
                                else:
                                    utils.unset_camera_config(camera_id)
                                    print(f"Deleted Camera: {camera_id}!")
                            else:
                                print(f"Deleted Camera: {camera_id}!")
                            time.sleep(2)
                        print("First Time Calling Start_Capture_Async")

                    message_to_send = {"message": f"message from {ws_url} server", "status_code": 2000, "video_process_server_id": ws_url, "ip_address": ip_address}
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
                            data = json.loads(message_received)
                            if 'status_code' in data and data['status_code'] == 4000:
                                print("Connection closed while msg sending")
                                await restart_websocket_connection(os.getenv("SERVER_ID"), fixed_server=True)
                                break
                            try:
                                threshold = float(data.get('threshold', os.getenv("THRESHOLD", 8.0)))
                                print(f"Received message: {message_received}")
                                if 'camera_running_status' in data:
                                    if data['camera_running_status']:
                                        print('going to start_capture_async')
                                        utils.set_camera_config(data.get('camera_url'), data.get('camera_id'), data.get('camera_type'), data.get('camera_running_status'), threshold)
                                        response = await utils.start_capture_async(data.get('camera_url'), data.get('camera_id'), data.get('camera_type'), threshold)
                                        response_json = json.dumps(response)
                                        await websocket.send(response_json)
                                    else:
                                        utils.set_camera_config(data.get('camera_url'), data.get('camera_id'), data.get('camera_type'), data.get('camera_running_status'), threshold)
                                        response = await utils.stop_capture_async(data.get('camera_id'))
                                        response_json = json.dumps(response)
                                        await websocket.send(response_json)
                                    
                                elif 'camera_check_url' in data:
                                    print('camera_check_url ', data.get('camera_check_url'))
                                    print(ws_url)
                                    response = await utils.cam_check_async(data.get('camera_check_url'), data.get('camera_type'), ws_url)
                                    
                                    response_json = json.dumps(response)
                                    await websocket.send(response_json)
                                
                                elif 'server_restart_trigger' in data:
                                    print('restarting server')
                                    await utils.restart_service()
                            except:
                                print("Error processing received message.")
                                continue
                    except websockets.ConnectionClosed:
                        if os.path.exists(config_location):
                            with open(config_location, 'r') as json_file:
                                data = json.load(json_file)
                                cameras = data.get('cameras', [])
                                for entry in cameras:
                                    camera_id = entry['camera_id']
                                    await utils.stop_capture_async(entry['camera_id'])
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
            await utils.stop_all_threads()
            await asyncio.sleep(10)
            
            #await restart_websocket_connection(os.getenv("SERVER_ID"), fixed_server=True)
            #break

async def on_startup(app):
    app['websocket_task'] = asyncio.ensure_future(connect_to_websocket(os.getenv("SERVER_ID"), fixed_server=True))

async def on_cleanup(app):
    if 'websocket_task' in app:
        app['websocket_task'].cancel()
        try:
            await app['websocket_task']
        except asyncio.CancelledError:
            print("WebSocket task cancelled successfully during cleanup.")

async def on_shutdown(app):
    # Cancel any background tasks
    if 'websocket_task' in app:
        app['websocket_task'].cancel()
        try:
            await app['websocket_task']
        except asyncio.CancelledError:
            print("WebSocket task cancelled successfully.shutdown")



app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)
app.on_shutdown.append(on_shutdown)

loop = asyncio.get_event_loop()
for sig in (signal.SIGINT, signal.SIGTERM):
    loop.add_signal_handler(sig, lambda s=sig: asyncio.ensure_future(shutdown(s, loop)))

app.router.add_get('/', index)
app.router.add_post('/', index)
app.router.add_static('/static', static_path)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=80)


async def shutdown(signal, loop):
    print(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.Task.all_tasks() if t is not asyncio.tasks.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()