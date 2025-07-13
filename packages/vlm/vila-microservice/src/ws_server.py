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

import asyncio
import logging
import uvicorn
from dataclasses import dataclass
from datetime import datetime
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from threading import Thread
from time import sleep


class Alert(BaseModel):
    title: str
    description: str
    alert_id: str
    type: str
    rule_string: str
    timestamp: str
    stream_id: str = ""

class WebSocketServer:

    def __init__(self, port=5016):
        self.port = port
        self.app = FastAPI()
        self.app.websocket("/v1/alerts/ws")(self.websocket_endpoint)

        self.active_connections = {}

    def send_alert(self, message):
        for con in self.active_connections.keys():
            logging.debug(f"Sending alert to {con} with data {message}")
            self.active_connections[con].put_nowait(message)
            logging.debug(f"placed data in {con} queue")
    async def websocket_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = asyncio.Queue()
        try:
            while True:
                alert = await self.active_connections[websocket].get()
                full_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                formatted_timestamp = datetime.strptime(full_timestamp, "%Y-%m-%d %H:%M:%S").strftime("%-I:%M%p")
                logging.debug("received alert")

                alert_obj = Alert(
                                    title = f"VLM Alert Triggered",
                                    description = f"The VLM has determined the alert rule: {alert['alert_str']} is True!",
                                    alert_id = str(alert["alert_id"]),
                                    type = "vlm_alert",
                                    rule_string = alert["alert_str"],
                                    timestamp = str(full_timestamp),
                                    stream_id = alert["stream_id"]
                                )

                logging.debug("sending alert over ws")
                await websocket.send_text(alert_obj.json())

        except Exception as e:
            logging.debug(e)
            del self.active_connections[websocket]

    def _start_server(self):
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

    def start_server(self):
        self.server_thread = Thread(target=self._start_server, daemon=True, name="REST Server")
        self.server_thread.start()

