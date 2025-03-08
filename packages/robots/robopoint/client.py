#!/usr/bin/env python3
# This example client can be run outside container, once the RoboPoint server is running.
# Install this in your environment first (these are already installed if running inside container)
#   pip install gradio_client opencv-python
import os
import time
import argparse
import cv2
import re
import base64

from gradio_client import Client, handle_file

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--server', type=str, default='http://0.0.0.0:7860/')
parser.add_argument('--model', type=str, default='robopoint-v1-vicuna-v1.5-13b')
parser.add_argument('--image', type=str, default=os.path.join(os.path.dirname(__file__), 'examples/sink.jpg'))
parser.add_argument('--camera', type=int, default=None, help='Camera device index (e.g., 0 for default camera)')
parser.add_argument('--request', type=str, required=True, help='Find free space in the sink to place a cup')

args = parser.parse_args()

print(f"{args}\n")
print(f"Connecting to RoboPoint server @ {args.server}")

# Connect to the server
client = Client(args.server)

# Load the model list
result = client.predict(
    api_name="/load_demo_refresh_model_list"
)

if args.camera is not None:
    print(f"Capturing image from camera device {args.camera}\n")
    print("Press Enter to capture and run inference")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            exit()
        cv2.imshow('Press Enter to capture', frame)
        if cv2.waitKey(1) == 13:  # Enter key
            break
    image_path = os.path.join(os.path.dirname(__file__), 'captured_image.jpg')
    cv2.imwrite(image_path, frame)
    cap.release()
    cv2.destroyAllWindows()
    args.image = image_path
else:
    print(f"Loading {args.image}\n")

# Load the request text from the CLI
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

# Extract the base64 image string
img_tag = result[0][1]
base64_str = re.search(r'data:image/jpeg;base64,(.*?)"', img_tag).group(1)

# Decode the base64 string to get the image
image_data = base64.b64decode(base64_str)

# Save the camera image to a file (optional)
with open("output_image.jpg", "wb") as f:
    f.write(image_data)

# Extract the coordinates
coordinates_str = re.search(r"\[(.*?)\]▌", img_tag).group(1)
coordinates = eval(coordinates_str)

print(f"Time:    {time.perf_counter() - time_begin:.2f} seconds")
print(f"2D Action Point Results: {coordinates}")
