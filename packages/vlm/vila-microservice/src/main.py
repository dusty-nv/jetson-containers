# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
import shutil
import time
from jetson_utils import cudaResize, cudaAllocMapped, cudaMemcpy
from mmj_utils.monitoring import AlertMonitor
from mmj_utils.overlay_gen import VLMOverlay
from mmj_utils.streaming import VideoOutput, VideoSource
from mmj_utils.vlm import VLM
from queue import Queue
from time import sleep

from api_server import VLMServer
from config import load_config
# from settings import load_config
from utils import process_query, vlm_alert_handler, vlm_chat_completion_handler
from ws_server import WebSocketServer


def find_config_path(base_env_var):
    config_path = os.environ[base_env_var]
    config_path_under_data_dir = os.environ[f"{base_env_var}_UNDER_DATA_DIR"]

    if os.path.exists(config_path_under_data_dir):
        return config_path_under_data_dir
    else:
        os.makedirs(os.path.dirname(config_path_under_data_dir), exist_ok=True)
        shutil.copy(config_path, config_path_under_data_dir)
        return config_path_under_data_dir

#Load config
config_path = find_config_path("MAIN_CONFIG_PATH")
config = load_config(config_path, "main")

logging.basicConfig(level=logging.getLevelName(config.log_level),
                    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

#Setup prometheus states and metric server
alertMonitor = AlertMonitor(config.max_alerts, config.prometheus_port, cooldown_time=config.alert_cooldown)

#Setup websocket server to send push alerts to mobile app
ws_server = WebSocketServer(port=config.websocket_server_port)
ws_server.start_server()

#create video input and output using mmj-utils,
v_input = VideoSource() #start as None will connect upon api request
v_output = VideoOutput(config.stream_output)

#create overlay object
overlay = VLMOverlay()

# Helper function to check if a message contains a base64 encoded image
def has_base64_image(message):
    """Check if a message contains a base64 encoded image."""
    if not hasattr(message, 'messages'):
        return False

    for msg in message.messages:
        if msg.role == "user" and isinstance(msg.content, list):
            for content in msg.content:
                if content.type == "image_url" and hasattr(content, "image_url"):
                    url = content.image_url.url
                    if url.startswith("data:image/"):
                        return True
    return False

#REST API Queue
cmd_q = Queue() #commands are put in by REST API requests and taken out by main loop
cmd_resp = dict() #responses to commands will be put in a shared dict based on command UUID

#Launch REST API server and connect queue to receive prompt and alert updates
api_server = VLMServer(cmd_q, cmd_resp, max_alerts=config.max_alerts, port=config.api_server_port)
api_server.start_server()

#Create VLM object
vlm = VLM(config.chat_server)
vlm.add_ego("alert", config.alert_system_prompt, vlm_alert_handler, {"alertMonitor": alertMonitor, "overlay":overlay, "ws_server":ws_server, "v_input":v_input})
vlm.add_ego("chat_completion", callback=vlm_chat_completion_handler, callback_args={"cmd_resp":cmd_resp, "overlay":overlay})

#Wait for VLM server to be loaded
while not vlm.health_check():
    if not cmd_q.empty():
        message = cmd_q.get()
        cmd_resp[message.id] = "VLM Model is still loading. Please try again later."
    sleep(5)
    logging.info("Waiting for VLM model health check")
logging.info("VLM Model health check passed. Starting main pipeline.")

#Start main pipeline
api_server_check = False #used to ensure api server queue is checked at least once between llm calls
active_message = None

#Setup buffer and timer to support multi-frame input to VLM
frame_buffer = [] #store frames for multi-frame input
frame_time_gap = config.multi_frame_input_time / config.multi_frame_input / 1000 #calc time between frames to pass to vlm in seconds
current_gap = frame_time_gap #track time between frames

while True:
    start_time = time.perf_counter() #track time between frames

    #First check controls from api server and update rules, queries or
    if not vlm.busy: #LLM is free, accept next command.
        api_server_check = True
        if not cmd_q.empty():
            message = cmd_q.get() #cmd from api server
            logging.debug("Received message from flask")

            if message.type == "alert":
                logging.debug("Message is an alert")

                alertMonitor.clear_alerts() #clear to stop old rules triggering alerts
                alertMonitor.set_alerts(message.data)
                overlay.input_text = message.data
                overlay.output_text = None
                active_message = message
                cmd_resp[message.id] = "Alert rules set"

            elif message.type == "query":
                logging.debug("Message is a query")

                #Clear old alerts
                alertMonitor.clear_alerts() #clear to stop old alerts triggering alerts

                # Check if the message contains a base64 encoded image
                has_image = has_base64_image(message.data)

                if not v_input.connected and not has_image:
                    # No stream and no base64 image, return error
                    cmd_resp[message.id] = "No stream has been added."
                    active_message = None
                else:
                    # Either stream is connected or message has base64 image
                    if has_image:
                        logging.info("Processing query with base64 encoded image")

                    overlay.input_text = process_query(message.data)
                    overlay.output_text = None
                    active_message = message

            elif message.type == "stream_add":
                logging.debug("Message is a stream add")

                # Properly close existing stream if connected
                if v_input.connected:
                    logging.warning("A registered stream exists. Closing before connecting to new stream.")
                    old_id = v_input.camera_id
                    v_input.close_stream()
                    logging.info(f"Closed existing stream with ID: {old_id}")

                # Connect to new stream
                stream_link = message.data["stream_url"]
                stream_id = message.data["stream_id"]
                logging.info(f"Connecting to new stream: {stream_link} with ID: {stream_id}")

                # Call connect_stream but don't check its return value (it always returns None)
                v_input.connect_stream(stream_link, camera_id=stream_id)

                # Check v_input.connected instead
                if v_input.connected:
                    logging.info(f"Successfully connected to stream: {stream_link}")
                    cmd_resp[message.id] = {"success": True, "message": "Successfully connected to stream"}
                else:
                    logging.error(f"Failed to connect to stream: {stream_link}")
                    cmd_resp[message.id] = {"success": False, "message": "Failed to connect to stream"}


            elif message.type == "stream_remove":
                logging.debug("Message is a stream remove")
                if v_input.connected:
                    if v_input.camera_id != message.data:
                        cmd_resp[message.id] = {"success": False, "message": f"Stream ID {message.data} does not exist. No stream removed."}
                    else:
                        v_input.close_stream()
                        cmd_resp[message.id] = {"success": True, "message": "Stream removed successfully"}
                        alertMonitor.clear_alerts() #clear to stop old alerts triggering alerts
                else:
                    cmd_resp[message.id] = {"success": False, "message": "No stream connected. Nothing to remove."}

            else:
                raise Exception("Received invalid message type from flask")

    #If a stream is added get frame and output
    if (frame := v_input()) is not None:
        frame_copy = cudaAllocMapped(width=frame.width, height=frame.height, format=frame.format)
        cudaMemcpy(frame_copy, frame)

        if current_gap >= frame_time_gap: #if enough time has passed, add a new frame to buffer
            if len(frame_buffer) >= config.multi_frame_input:
                frame_buffer.pop(0) #pop oldest if buffer is full
            frame_buffer.append(frame_copy) #add latest frame
            current_gap = 0 #reset timer

        if not vlm.busy and active_message and api_server_check: #llm available & query or alerts
            if active_message.type == "alert":
                logging.info("Making VLM call with alert input")
                vlm("alert", active_message.data, copy.copy(frame_buffer)) #send shallow copy of frame buffer to avoid race conditions
            elif active_message.type == "query":
                logging.info("Making VLM call with query input")
                vlm("chat_completion", active_message.data, copy.copy(frame_buffer), message_id=message.id)
                active_message = None #only send queries 1 time
            else:
                logging.error(f"Message type is invalid: {active_message.type}")

            api_server_check = False

        #resize frame to 1080p for smooth output
        resized_frame = cudaAllocMapped(width=1920, height=1080, format=frame.format)
        cudaResize(frame, resized_frame)

        #generate overlay
        resized_frame = overlay(resized_frame)
        v_output(resized_frame)
    # Handle the case where there's no stream but we have a query with a base64 image
    elif not vlm.busy and active_message and api_server_check and active_message.type == "query" and has_base64_image(active_message.data):
        logging.info("Making VLM call with base64 image (no stream)")
        # Pass None or empty list as frame buffer since we don't need it for base64 images
        vlm("chat_completion", active_message.data, None, message_id=message.id)
        active_message = None
        api_server_check = False

    else:
        sleep(1/30)

    loop_time = time.perf_counter() - start_time
    #logging.info(f"Main Loop Time: {loop_time}")
    current_gap += loop_time
