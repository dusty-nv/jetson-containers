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

import logging
import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from queue import Queue
from threading import Thread
from time import sleep, time
from typing import Union, List
from uuid import uuid4

from .api_schemas import StreamAdd, Response, StreamMeta


@dataclass
class APIMessage:
    type: str
    data: Union[str, dict]
    id: int
    time: int

class APIServer:
    """Base API Server that implements health, and live-stream endpoints. Also includes logic for handling inter-process communication."""
    def __init__(self, cmd_q, resp_d, port=5000, clean_up_time=180):
        """
        Initialize API Server class. Will not start the server. Call start_server() after initializing.

        cmd_q: APIServer places command in this queue. Main thread consumes these commands and returns responses in the resp_d.
        resp_d: Main thread returns responses to this shared dictionary. APIServer will respond to the requests based on this dictionary
        port: Port the API server will run on
        clean_up_time: Time to clean up any responses that were added after a time out. Ensures resp_d will not grow unbounded.
        """
        self.cmd_q = cmd_q #send commands out
        self.resp_d = resp_d #track responses
        self.clean_up_time = clean_up_time
        self.cmd_tracker = {} #store cmd uuid and timestamps

        self.port = port

        self.app = FastAPI()

        #CORS setup for development.
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.added_stream = None

        # Removed live-stream endpoints to prevent them from being exposed
        # but kept all the stream-related functions for internal use
        self.app.get("/api/v1/health/ready")(self.health)
        self.app.head("/api/v1/health/ready")(self.health)

    def get_command_response(self, uuid_str, timeout=60):
        """Logic to wait for a command response from another thread on the resp_d."""
        start_time = time()

        #remove any old responses
        resp_d_keys = list(self.resp_d.keys())
        for k in resp_d_keys:
            if start_time - self.cmd_tracker[k] >= self.clean_up_time:
                self.resp_d.pop(k)
                self.cmd_tracker.pop(k)

        #wait for response or timeout
        while True:
            if time() - start_time >= timeout:
                logging.debug("Timed out waiting for command response")
                return None

            if uuid_str in self.resp_d:
                logging.debug(f"Received response on id {uuid_str}")
                self.cmd_tracker.pop(uuid_str)
                return self.resp_d.pop(uuid_str)
            else:
                sleep(0.1)

    def health(self):
        """Health endpoint"""
        return Response(detail="healthy")

    def stream_add(self, body: StreamAdd):
        """Add stream from live-stream endpoint """
        stream_url = body.liveStreamUrl
        stream_id = str(uuid4()).strip()  # Always generate stream_id

        data = {"stream_id": stream_id, "stream_url": stream_url}
        queue_message = APIMessage(type="stream_add", data=data, id=str(uuid4()), time=time())
        self.cmd_q.put(queue_message)
        self.cmd_tracker[queue_message.id] = queue_message.time

        response = self.get_command_response(queue_message.id)  # TODO make timeout an option set by client
        if response:
            if response["success"]:
                self.added_stream = StreamMeta(id=stream_id, liveStreamUrl=stream_url, description=body.description)
                return {"id": stream_id}
            else:
                raise HTTPException(status_code=422, detail=response["message"])
        else:
            raise HTTPException(status_code=500, detail="Server timed out processing the request")

    def stream_list(self):
        """List added stream to live-stream endpoint"""
        if self.added_stream:
            return [self.added_stream]
        else:
            return []

    def stream_remove(self, stream_id: str):
        """Remove added streams from live-stream endpoint"""
        queue_message = APIMessage(type="stream_remove", data=stream_id.strip(), id=str(uuid4()), time=time())
        self.cmd_q.put(queue_message)
        self.cmd_tracker[queue_message.id] = queue_message.time
        response = self.get_command_response(queue_message.id)
        if response:
            if response["success"]:
                self.added_stream = None
                return Response(detail=response["message"])
            else:
                raise HTTPException(status_code=422, detail=response["message"])
        else:
            raise HTTPException(status_code=500, detail="Server timed out processing the request")

    def _start_server(self):
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

    def start_server(self):
        """Call this function to launch the server."""
        self.server_thread = Thread(target=self._start_server, daemon=True, name="REST Server")
        self.server_thread.start()
