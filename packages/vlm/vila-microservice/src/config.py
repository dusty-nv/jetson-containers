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
from pydantic import conint
from pydantic_settings import BaseSettings


class MainConfig(BaseSettings):
    api_server_port: int
    prometheus_port: int
    websocket_server_port: int
    stream_output: str
    chat_server: str
    alert_system_prompt: str
    max_alerts: int
    alert_cooldown: int
    log_level: str
    multi_frame_input: conint(ge=1, le=8)
    multi_frame_input_time: conint(ge=0)

class ChatServerConfig(BaseSettings):
    api_server_port:int
    prometheus_port: int
    print_stats: bool
    log_level: str
    model: str

def load_config(config_path, type):
    with open(config_path, 'r') as file:
        config_data = json.load(file)

    if type == "main":
        return MainConfig(**config_data)
    elif type == "chat_server":
        return ChatServerConfig(**config_data)
