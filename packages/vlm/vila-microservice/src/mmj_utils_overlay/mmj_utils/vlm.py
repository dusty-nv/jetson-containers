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

import base64
import io
import logging
import requests
from PIL import Image
from jetson_utils import cudaToNumpy
from threading import Thread

from .api_schemas import ChatContentImage, ChatContentImageOptions, ChatMessages


class VLM:
    """Abstraction for the VLM chat server. Can use this object to track different 'egos' that have an associated system prompt and callback function."""

    def __init__(self, url):
        self.url = url + "/v1/chat/completions"
        self.alert_url = url + "/v1/alert/completions"
        self.health_url = url + "/v1/health"
        self.egos = {}
        self.busy = False

    def health_check(self):
        """Check health endpoint of the chat server."""
        try:
            response = requests.get(self.health_url)
            if not response.ok:
                return False

            if response.json() == {"detail": "ready"}:
                return True
            else:
                return False
        except requests.RequestException as e:
            logging.info("Error occurred in health check: {e}")
            return False


    def _encode_image(self, image):
        """Encodes input image as b64 for embedding in chat server request."""
        #accepts cudaImage
        image = cudaToNumpy(image)
        image = Image.fromarray(image)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image = buf.getvalue()
        image = base64.b64encode(image).decode("utf-8")
        logging.info("Encoded image")
        return image

    def add_ego(self, name, system_prompt="You are a helpful AI assistant.", callback=None, callback_args=None, history=None):
        """Add a new ego with an associated system prompt and callback function"""
        self.egos[name] = {"system_prompt": system_prompt, "callback":callback, "callback_args":callback_args}

    def replace_images(self, chat_completion, images):
        """Replace all stream type content with the base64 encoded frame and v4l2 URLs with actual images"""
        image_counter = 0
        for m, message in enumerate(chat_completion.messages):
            if isinstance(message.content, str):
                continue
            elif isinstance(message.content, list):
                for c, content in enumerate(message.content):
                    # Handle stream type content
                    if content.type == "stream":
                        if images is None or image_counter >= len(images):
                            logging.info("Not enough images were supplied with the chat completion prompt. Skipping stream content replacement.")
                            del chat_completion.messages[m].content[c]
                            continue
                        while image_counter < len(images):
                            insert_counter = c
                            image = images[image_counter]
                            image = self._encode_image(image)
                            image_string = f"data:image/jpeg;base64,{image}"
                            image_content = ChatContentImage(type="image_url", image_url=ChatContentImageOptions(url=image_string))
                            chat_completion.messages[m].content[insert_counter:insert_counter] = [image_content]
                            image_counter += 1
                            insert_counter += 1
                        del chat_completion.messages[m].content[c]

                    # Handle custom video sources
                    elif content.type == "image_url" and hasattr(content, "image_url"):
                        url = content.image_url.url
                        # Check if it's a custom video source
                        CUSTOM_VIDEO_SCHEMES = ("csi://", "v4l2://", "webrtc://", "rtp://", "rtsp://", "file://")
                        if any(url.startswith(scheme) for scheme in CUSTOM_VIDEO_SCHEMES):
                            if images is None or image_counter >= len(images):
                                logging.info("Not enough images were supplied for custom video source replacement.")
                                continue

                            # Replace custom video source URL with base64 encoded image
                            image = images[image_counter]
                            image = self._encode_image(image)
                            image_string = f"data:image/jpeg;base64,{image}"
                            chat_completion.messages[m].content[c].image_url.url = image_string
                            image_counter += 1
            else:
                logging.info("Invalid content format")

        return chat_completion

    def _call_chat_completion(self, ego, chat_completion, images, callback_args={}):
        """Calls chat server chat completions object followed up by the associated ego callback function"""
        try:
            chat_completion = self.replace_images(chat_completion, images)
            response = requests.get(self.url, json=chat_completion.dict())
            logging.debug(response.text)
            response = response.json()

            if (callback:=self.egos[ego]["callback"]):
                callback_args.update(self.egos[ego]["callback_args"])
                callback(response, **callback_args) #unpack dict of args as kwargs
        except Exception as e:
            logging.error(e)
            self.busy = False
        self.busy = False

    def _call_alert_completion(self, ego, message, images, callback_args={}):
        """Calls optimized endpoint for batch alert completion"""
        try:
            image_strings = []
            user_prompts = [x for x in message.values()]
            for image in images:
                image_string = self._encode_image(image)
                image_strings.append(f"data:image/jpeg;base64,{image_string}")
            payload = {"images":image_strings, "system_prompt":self.egos[ego]["system_prompt"], "user_prompts":user_prompts, "min_tokens":1, "max_tokens":1}
            response = requests.get(self.alert_url, json=payload)
            logging.debug(response.text)
            response = response.json()

            if (callback:=self.egos[ego]["callback"]):
                callback_args.update(self.egos[ego]["callback_args"])
                callback(response, **callback_args) #unpack dict of args as kwargs

        except Exception as e:
            logging.error(e)
            self.busy = False
        self.busy = False

    def __call__(self, ego, message, images=None, **kwargs):
        """Call the VLM with the ego. Message can be a simple string prompt and image or a Chat Completion object."""
        if self.busy:
            logging.info("VLM is busy")
            return None

        if ego not in self.egos:
            logging.info("Ego does not exist. Add it first.")
            return None

        if not isinstance(message, (dict, ChatMessages)):
            logging.info("Message is an unsupported type")
            return None

        else:
            self.busy = True
            if isinstance(message, ChatMessages):
                Thread(target=self._call_chat_completion, args=(ego, message, images, kwargs)).start() #pass kwargs as dict to callback
            if isinstance(message, dict):
                Thread(target=self._call_alert_completion, args=(ego, message, images, kwargs)).start() #pass kwargs as dict to callback
