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

import json
import logging
import os
import re
import requests
from fastapi import HTTPException
from mmj_utils.api_schemas import ChatMessages, StreamAdd
from mmj_utils.api_server import APIServer, APIMessage, Response
from pydantic import BaseModel, Field, conlist, constr
from time import time
from typing import Optional, Dict, Any
from uuid import uuid4

from config import load_config


class Alerts(BaseModel):
    alerts: conlist(item_type=constr(max_length=100), min_length=0, max_length=10)
    id: Optional[constr(max_length=100)] = ""

class Query(BaseModel):
    query: str = Field(..., max_length=20, min_length=0)

class VLMServer(APIServer):
    def __init__(self, cmd_q, resp_d, port=5000, clean_up_time=180, max_alerts=10):
        super().__init__(cmd_q, resp_d, port=port, clean_up_time=clean_up_time)

        self.max_alerts = max_alerts

        self.app.post("/v1/alerts")(self.alerts)
        self.app.post("/v1/chat/completions")(self.chat_completion)
        self.app.get("/v1/models")(self.get_model_name)

    def get_model_name(self):
        """
        Returns the current model name.
        """
        #Load config
        chat_server_config_path = os.environ["CHAT_SERVER_CONFIG_PATH"]
        chat_server_config = load_config(chat_server_config_path, "chat_server")

        model_name = chat_server_config.model
        current_timestamp = int(time())  # Substitute by current timestamp

        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": current_timestamp,
                    "owned_by": "system"
                }
            ]
        }

    def alerts(self, body:Alerts):
        user_alerts = dict()

        #parse input alerts
        for x, alert in enumerate(body.alerts):
            if x >= self.max_alerts:
                break

            alert = alert.strip()

            if alert != "":
                user_alerts[f"r{x}"] = alert

        message_data = user_alerts
        queue_message = APIMessage(type="alert", data=message_data, id=str(uuid4()), time=time())

        self.cmd_q.put(queue_message)
        self.cmd_tracker[queue_message.id] = queue_message.time

        response = self.get_command_response(queue_message.id)
        if response:
            return Response(detail=response)
        else:
            raise HTTPException(status_code=500, detail="Server timed out processing the request")

    def chat_completion(self, body: ChatMessages):
        # Check if there's a v4l2 URL in the request
        custom_video_source = None
        image_url = None
        stream_id = None

        # Define regex for custom video sources
        HTTP_SCHEMES = ("http://", "https://")  # OpenAI Vision API standard image URLs
        CUSTOM_VIDEO_SCHEMES = ("csi://", "v4l2://", "webrtc://", "rtp://", "rtsp://", "file://")

        # Look for v4l2 URLs in the messages
        for message in body.messages:
            if message.role == "user" and isinstance(message.content, list):
                for content in message.content:
                    if content.type == "image_url" and hasattr(content, "image_url"):
                        url = content.image_url.url

                        # Determine the URL type
                        if url.startswith(HTTP_SCHEMES):
                            image_url = url  # This will later be handled (download & register)
                            logging.info(f"Found standard image URL: {image_url}")
                        elif url.startswith(CUSTOM_VIDEO_SCHEMES):
                            custom_video_source = url
                            logging.info(f"Found custom video source: {custom_video_source}")
                        elif url.startswith("data:image/"):
                            logging.info(f"Found base64 encoded image")

        # If an image URL is found, handle downloading and registering as file://
        if image_url:

            save_dir = "/data/images"  # Define storage path
            os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

            # Generate a unique filename
            filename = f"image_{uuid4().hex}.jpg"
            save_path = os.path.join(save_dir, filename)

            try:
                logging.info(f"Downloading image from {image_url}...")
                response = requests.get(image_url, timeout=10)

                if response.status_code == 200:
                    with open(save_path, "wb") as file:
                        file.write(response.content)
                    logging.info(f"Image saved to {save_path}")
                    custom_video_source = f"file://{save_path}"
                else:
                    logging.error(f"Failed to download image {image_url}: HTTP {response.status_code}")
            except Exception as e:
                logging.error(f"Error downloading image {image_url}: {str(e)}")

        # If a custom video source is found, handle it accordingly
        if custom_video_source:
            # Check if there's already a stream connected
            existing_streams = self.stream_list()
            stream_already_connected = False

            for stream in existing_streams:
                if stream.liveStreamUrl == custom_video_source:
                    # Stream with this device is already connected, use it
                    stream_id = stream.id
                    stream_already_connected = True
                    logging.info(f"Using existing v4l2 stream with ID: {stream_id}")
                    break

            # If no stream is connected with this device, add a new one
            if not stream_already_connected:
                # Create a StreamAdd object
                stream_add_body = StreamAdd(liveStreamUrl=custom_video_source, description="custom video source")

                try:
                    # Call the stream_add function directly
                    stream_result = self.stream_add(stream_add_body)
                    logging.debug(f"Stream add result: {stream_result}, type: {type(stream_result)}")

                    # Handle the case where stream_result might be a string or a dictionary
                    if isinstance(stream_result, dict) and "id" in stream_result:
                        stream_id = stream_result["id"]
                    elif isinstance(stream_result, str):
                        # If it's a string, it might be a JSON string
                        try:
                            result_dict = json.loads(stream_result)
                            if isinstance(result_dict, dict) and "id" in result_dict:
                                stream_id = result_dict["id"]
                            else:
                                raise ValueError("Invalid JSON format")
                        except json.JSONDecodeError:
                            # If it's not a JSON string, use it directly as the stream_id
                            stream_id = stream_result
                    else:
                        raise ValueError(f"Unexpected stream_result type: {type(stream_result)}")

                    logging.info(f"Added custom video stream with ID: {stream_id}")
                except Exception as e:
                    logging.error(f"Failed to add custom video stream: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Failed to add custom video stream: {str(e)}")

        # Process the chat completion request
        queue_message = APIMessage(type="query", data=body, id=str(uuid4()), time=time())
        self.cmd_q.put(queue_message)
        self.cmd_tracker[queue_message.id] = queue_message.time

        response = self.get_command_response(queue_message.id)

        # Keep the stream connected after processing
        # (removed the stream removal code to maintain the connection)

        if response:
            return response
        else:
            raise HTTPException(status_code=500, detail="Server timed out processing the request")
