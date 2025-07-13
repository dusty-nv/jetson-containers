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

from pydantic import BaseModel, conlist, constr, conint
from typing import Optional, Union, Dict, List, Literal

#Optimized Alert Schemea
class AlertCompletion(BaseModel):
    """Schema for batch alert completions"""
    system_prompt: Optional[constr(min_length=0, max_length=20000)]
    images: conlist(item_type=constr(min_length=1, max_length=5000000, pattern=r'^data:image/[a-zA-Z]+;base64,.*'), min_length=1)
    user_prompts: conlist(item_type=constr(min_length=0, max_length=20000), min_length=1)
    max_tokens: Optional[conint(gt=0, le=4096)] = 128
    min_tokens: Optional[conint(gt=0, le=4096)] = 1

class AlertCompletionResult(BaseModel):
    """Schema for batch alert completions"""
    alert_response: conlist(item_type=constr(min_length=0, max_length=250), min_length=1)


#Streaming Schemas
class StreamMeta(BaseModel):
    """Schema for Stream content metadata"""
    id: Optional[constr(min_length=0, max_length=100)] = None
    liveStreamUrl: Optional[constr(min_length=0, max_length=1000)] = None
    description: Optional[constr(min_length=0, max_length=1000)] = None

class StreamAdd(BaseModel):
    """Schema for adding a stream"""
    liveStreamUrl: constr(min_length=0, max_length=1000)
    description: Optional[constr(min_length=0, max_length=1000)] = ""

class StreamAddResponse(BaseModel):
    """Response schema for adding a stream"""
    id: constr(min_length=0, max_length=100)

#Chat Completion Schemas
class Response(BaseModel):
    """Schema for generic response"""
    detail: constr(min_length=0, max_length=1000)

class ChatContentImageOptions(BaseModel):
    """Schema for b64 encoded image content or video device URL"""
    url: constr(
        min_length=0,
        max_length=5000000,
        pattern=r'^(data:image/[a-zA-Z]+;base64,.*|https?://.*|csi://.*|v4l2://.*|webrtc://.*|rtp://.*|rtsp://.*|file://.*)'
    ) #5MB max b64 string or custom video source

class ChatContentStreamOptions(BaseModel):
    """Schema for stream content ID"""
    stream_id: constr(min_length=0, max_length=1000)

class ChatContentImage(BaseModel):
    """Schema for image content"""
    type: Literal['image_url']
    image_url: ChatContentImageOptions

class ChatContentStream(BaseModel):
    """Schema for stream content"""
    type: Literal['stream']
    stream: ChatContentStreamOptions

class ChatContentText(BaseModel):
    """Schema for text content"""
    type: Literal['text']
    text: constr(min_length=0, max_length=20000)

class ChatMessage(BaseModel):
    """Schema for a chat message"""
    role: Literal['system', 'user', 'assistant']
    content: Union[str, List[Union[ChatContentText, ChatContentImage, ChatContentStream]]]

class ChatMessages(BaseModel):
    """Schema for chat completions"""
    messages: conlist(item_type=ChatMessage, min_length=1, max_length=100)
    max_tokens: Optional[conint(gt=0, le=4096)] = 128
    min_tokens: Optional[conint(gt=0, le=4096)] = 1

class ChatCompletion(BaseModel):
    """Schema for chat completions response"""
    index: conint(gt=-1, le=4096)
    message: ChatMessage

class ChatCompletions(BaseModel):
    """Schema for returned chat completions message"""
    choices: conlist(item_type=ChatCompletion, min_length=1, max_length=4096)
