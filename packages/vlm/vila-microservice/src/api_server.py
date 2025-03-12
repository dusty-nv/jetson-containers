# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from uuid import uuid4
from time import time 

from mmj_utils.api_server import APIServer, APIMessage, Response 
from mmj_utils.api_schemas import ChatMessages

from pydantic import BaseModel, Field, conlist, constr 
from typing import Optional 
from fastapi import HTTPException 


class Alerts(BaseModel):
    alerts: conlist(item_type=constr(max_length=100), min_length=0, max_length=10)
    id: Optional[constr(max_length=100)] = ""

class Query(BaseModel):
    query: str = Field(..., max_length=20, min_length=0)

class VLMServer(APIServer):
    def __init__(self, cmd_q, resp_d, port=5000, clean_up_time=180, max_alerts=10):
        super().__init__(cmd_q, resp_d, port=port, clean_up_time=clean_up_time)
        
        self.max_alerts = max_alerts

        self.app.post("/api/v1/alerts")(self.alerts)
        self.app.post("/api/v1/chat/completions")(self.chat_completion)

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
        queue_message = APIMessage(type="query", data=body, id=str(uuid4()), time=time())
        self.cmd_q.put(queue_message)
        self.cmd_tracker[queue_message.id] = queue_message.time

        response = self.get_command_response(queue_message.id)
        if response:
            return response
        else:
            raise HTTPException(status_code=500, detail="Server timed out processing the request")