#!/usr/bin/env python3
# This example client can be run outside container, once the RoboPoint server is running.
# Install this in your environment first (these are already installed if running inside container)
#   pip install gradio_client
import os
import time
import argparse

from gradio_client import Client, handle_file

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--server', type=str, default='http://0.0.0.0:7860/')
parser.add_argument('--model', type=str, default='robopoint-v1-vicuna-v1.5-13b')
parser.add_argument('--image', type=str, default=os.path.join(os.path.dirname(__file__), 'examples/sink.jpg'))
parser.add_argument('--request', type=str, required=True)

args = parser.parse_args()

print(f"{args}\n")
print(f"Connecting to RoboPoint server @ {args.server}")

#connect to the server
client = Client(args.server)

#load the model list
result = client.predict(
    api_name="/load_demo_refresh_model_list"
)

print(f"Loading {args.image}\n")

#load the request text from the cli
result = client.predict(
    text=args.request,
    image=handle_file(args.image),
    image_process_mode="Pad",
    api_name="/add_text_1"
)

print(f"Generating output with {args.model}\n")
time_begin = time.perf_counter()

# generate the 2D action point prediction
result = client.predict(
    model_selector=args.model,
    temperature=1,
    top_p=0.7,
    max_new_tokens=512,
    api_name="/http_bot_2"
)

result = result[-1][-1]
result = result[result.rfind('[('):]

print(f"Time:    {time.perf_counter() - time_begin:.2f} seconds")
print(f"Results: {result}")
