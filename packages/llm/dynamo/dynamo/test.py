#!/usr/bin/env python3
print('testing dynamo...')


import argparse
import asyncio
import sys
import time

import uvloop
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.openai_api.adapter import v1_chat_generate_request
from sglang.srt.openai_api.protocol import ChatCompletionRequest
from sglang.srt.server_args import ServerArgs

from dynamo.llm import ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker

DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
print('dynamo OK\n')