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
import os
import shutil
import uvicorn
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI
from mmj_utils.api_schemas import *
from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import print_table
from prometheus_client import start_http_server, Gauge
from threading import Thread
from time import time

from config import load_config


def decode_image(base64_string):
    start = time()

    # Decode the base64 string
    image_data = base64.b64decode(base64_string)

    # Convert binary data to an image
    image = Image.open(io.BytesIO(image_data))

    # Convert RGBA to RGB (fix for torchvision normalize issue)
    if image.mode == "RGBA":
        logging.warning("Image is RGBA, converting to RGB...")
        image = image.convert("RGB")

    end = time()
    logging.info(f"Image decoding time: {end-start} seconds")

    return image

model = None
chat_history = None
model_loaded = False
model_stats = {}

def load_model(model_name):
    global model
    global model_loaded
    global model_stats
    global chat_history
    model = NanoLLM.from_pretrained(config.model, api="mlc", quantization="q4f16_ft", vision_api="hf", print_stats=True)
    #warm up
    chat_history = ChatHistory(model, system_prompt="you are a helpful AI assistant.")
    chat_history.append(role="user", text="what are you capable of?")
    embedding, _ = chat_history.embed_chat()
    reply = model.generate(
            embedding,
            streaming=False,
            max_new_tokens=128,
            min_new_tokens=1
        )
    chat_history.reset()
    logging.debug(f"Warm up reply: {reply}")

    for key in model.stats:
        model_stats[key] = Gauge(key, key)
    model_loaded = True

#Load model on startup
@asynccontextmanager #https://fastapi.tiangolo.com/advanced/events/
async def lifespan(app: FastAPI):
    start_http_server(config.prometheus_port) #start prometheus metric server
    model_load_thread = Thread(target=load_model, args=(config.model, ))
    model_load_thread.start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/v1/health")
def health():
    if model_loaded:
        return Response(detail="ready")
    else:
        return Response(detail="model loading")

@app.get("/v1/alert/completions")
async def request_alert_completions(body: AlertCompletion):
    """Applies a batch of text prompts across the same set of input images for alerts"""
    global chat_history
    if not model_loaded:
        return {"choices":[]}

    results = []

    #add system prompt
    chat_history.reset()
    chat_history.append("system", body.system_prompt)

    #add images
    for image_url in body.images:
        # Extract base64 data after the comma
        image_string = image_url.split(',', 1)[1]
        image = decode_image(image_string)
        chat_history.append("user", image=image)

    #add user prompt and inference
    for user_prompt in body.user_prompts:
        start = time()
        chat_history.append("user", user_prompt)

        #inference on model
        embedding, _ = chat_history.embed_chat()
        print(len(chat_history))
        reply = model.generate(
                embedding,
                streaming=True,
                kv_cache=chat_history.kv_cache,
                max_new_tokens=body.max_tokens,
                min_new_tokens=body.min_tokens
            )
        str_reply = ""
        for token in reply:
            if reply.eos:
                break
            else:
                str_reply+=token

        chat_history.append("bot", reply)
        chat_history.pop(2)
        str_reply = str_reply.replace("\n", "").replace("</s>", "").strip()
        #pop bot reply and user prompt
        end = time()
        print(f"Alert request time: {end-start} s")
        results.append(str_reply)
        print(results)
    chat_history.reset()
    return AlertCompletionResult(alert_response=results)


@app.get("/v1/chat/completions")
async def request_chat_completion(body: ChatMessages):
    global chat_history

    if not model_loaded:
        return {"choices":[]}

    start = time()

    system_prompt = "You are a helpful assistant."
    if body.messages[0].role == "system":
        system_prompt = body.messages[0].content
        logging.info(f"Received system prompt: {system_prompt}")

    chat_history.reset()
    chat_history.append("system", system_prompt)
    for message in body.messages:
        if message.role == "user":
            if isinstance(message.content, str):
                chat_history.append(role="user", text=message.content)

            elif isinstance(message.content, list):
                for content in message.content:
                    if content.type == "text":
                        chat_history.append(role="user", text=content.text)
                        logging.info(f"adding user text: {content.text}")
                    elif content.type == "image_url":
                        url = content.image_url.url
                        if url.startswith("data:image/"):
                            # Extract base64 data after the comma
                            image_string = url.split(',', 1)[1]
                            image = decode_image(image_string)
                            chat_history.append(role="user", image=image)
                            logging.info(f"added image")
                        else:
                            # Skip non-base64 URLs (they should have been handled by the API server)
                            logging.info(f"Skipping non-base64 URL: {url[:10]}...")

            else:
                logging.info(f"Message is invalid type: {type(message.content)}")

        elif message.role == "assistant":
            chat_history.append(role="bot", text=message.content)

        elif message.role == "system": #only take into account the system prompt if first message
            continue

        else:
            logging.info(f"Unsupported role type: {message.role}")

    embedding, _ = chat_history.embed_chat()

    reply = model.generate(
            embedding,
            streaming=False,
            max_new_tokens=body.max_tokens,
            min_new_tokens=body.min_tokens
        )
    reply = reply.replace("\n", "").replace("</s>", "").strip()
    stop = time()
    logging.info(f"Server processing time: {stop-start} seconds")
    for key in model.stats:
        model_stats[key].set(model.stats[key])

    if config.print_stats:
        print_table(model.stats)
    chat_history.reset()
    return ChatCompletions(choices=[ChatCompletion(index=0, message=ChatMessage(role="assistant", content=reply))])


# @app.get("/v1/models")
# def get_model_name():
#     """
#     Returns the current model name.
#     """
#     return {"model": config.model}

def find_config_path(base_env_var):
    config_path = os.environ[base_env_var]
    config_path_under_data_dir = os.environ[f"{base_env_var}_UNDER_DATA_DIR"]

    if os.path.exists(config_path_under_data_dir):
        return config_path_under_data_dir
    else:
        os.makedirs(os.path.dirname(config_path_under_data_dir), exist_ok=True)
        shutil.copy(config_path, config_path_under_data_dir)
        return config_path_under_data_dir

if __name__ == "__main__":
    #Load config
    config_path = find_config_path("CHAT_SERVER_CONFIG_PATH")
    config = load_config(config_path, "chat_server")

    logging.basicConfig(level=logging.getLevelName(config.log_level),
                    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

    model_name = config.model.split("/")[-1]
    if not os.path.exists(f"/data/models/mlc/dist/models/{model_name}"):
        logging.info("Model path not found. Will download and build model. This will take some time.")
        os.makedirs("/data/models/mlc/dist/models", exist_ok=True)

    uvicorn.run(app, host="0.0.0.0", port=config.api_server_port)
