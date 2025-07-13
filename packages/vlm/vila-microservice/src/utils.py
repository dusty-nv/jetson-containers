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
import re


#process alerts reply from llm
def process_alerts_reply(reply):
    try:
        formatted_reply = dict()
        for i, r in enumerate(reply):
            if r.lower().strip() in ["yes", "1", "true"]:
                formatted_reply[f"r{i}"] = True
            else:
                formatted_reply[f"r{i}"] = False

    except Exception as e:
        logging.debug(e)
        logging.info(f"LLM Reply could not be parsed: {reply}")
        formatted_reply = dict()

    return formatted_reply


def process_query(query):
    for message in query.messages:
        if message.role == "user":
            if isinstance(message.content, str):
                return message.content
            # if not string should be list
            for content in message.content:
                if content.type == "text":
                    return content.text
    return "No user prompt found."


def vlm_alert_handler(response, **kwargs):
    try:
        logging.info("Updating prometheus alerts")
        response = response["alert_response"]
        v_input = kwargs["v_input"]

        #when alert states change, update prometheus and overlay
        alertMonitor = kwargs["alertMonitor"]
        alert_states = process_alerts_reply(response)
        logging.info(f"Updated alert states: {alert_states}")


        alert_cooldowns = {key: alertMonitor.alerts[key].cooldown for key in alertMonitor.alerts}
        alertMonitor.set_alert_states(alert_states)

        ws_server = kwargs["ws_server"]

        for alert_id, alert in alertMonitor.alerts.items():
            logging.info(alert_id)
            alert_string = alert.string
            alert_state = alert.state
            logging.info(alert_state)
            if alert_state == 1 and not alert_cooldowns[alert_id]:
                data = {"stream_url":v_input.url, "stream_id":v_input.camera_id, "camera_name":v_input.camera_name, "alert_id":alert_id, "alert_str":alert_string}
                logging.info(data)
                ws_server.send_alert(data)

        logging.info("updating overlay")
        overlay = kwargs["overlay"]
        overlay.output_text = str(alert_states)
        overlay.reset_decay()
    except Exception as e:
        logging.info(e)

def vlm_chat_completion_handler(response, **kwargs):
    overlay = kwargs["overlay"]
    cmd_resp = kwargs["cmd_resp"]
    message_id = kwargs["message_id"]
    logging.info("Sending query response")
    cmd_resp[message_id] = response
    overlay.output_text = response["choices"][0]["message"]["content"]
    overlay.reset_decay()
