#!/usr/bin/env python3
import os
import sys
import time
import datetime
import resource
import argparse
import socket

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from nanosam.utils.predictor import Predictor

parser = argparse.ArgumentParser()
parser.add_argument("--image_encoder", type=str, default="/opt/nanosam/data/resnet18_image_encoder.engine")
parser.add_argument("--mask_decoder", type=str, default="/opt/nanosam/data/mobile_sam_mask_decoder.engine")

parser.add_argument('-i', '--images', action='append', nargs='*', help="Paths to images to test")

parser.add_argument('-r', '--runs', type=int, default=2, help="Number of inferencing runs to do (for timing)")
parser.add_argument('-w', '--warmup', type=int, default=1, help='the number of warmup iterations')
parser.add_argument('-s', '--save', type=str, default='', help='CSV file to save benchmarking results to')
   
args = parser.parse_args()

if not args.images:
    args.images = [
        "/opt/nanosam/assets/dogs.jpg",
        "/data/images/hoover.jpg",
        "/data/images/lake.jpg",
    ]

print(args)

def get_max_rss():  # peak memory usage in MB (max RSS - https://stackoverflow.com/a/7669482)
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  

# Instantiate TensorRT predictor
predictor = Predictor(
    args.image_encoder,
    args.mask_decoder
)

avg_encoder=0
avg_latency=0
pil_image=None
mask=None

for run in range(args.runs + args.warmup):
        
    for image in args.images:

        time_begin=time.perf_counter()

        # Read image and run image encoder
        pil_image = PIL.Image.open(image)
        predictor.set_image(pil_image)

        time_encoder=time.perf_counter() - time_begin
        
        print(f"{image}") 
        print(f"  encode        :  {time_encoder:.3f} seconds")

        # Segment using bounding box
        bbox = [100, 100, 850, 759]  # x0, y0, x1, y1

        points = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]]
        ])

        point_labels = np.array([2, 3])

        mask, _, _ = predictor.predict(points, point_labels)

        time_elapsed=time.perf_counter() - time_begin

        print(f"  full pipeline :  {time_elapsed:.3f} seconds\n")

        if run >= args.warmup:
            avg_encoder += time_encoder
            avg_latency += time_elapsed
        
    
avg_encoder /= ( args.runs * len(args.images) )
avg_latency /= ( args.runs * len(args.images) )

memory_usage=get_max_rss()

print(f"AVERAGE of {args.runs} runs:")
print(f"  encoder --- {avg_encoder:.3f} sec")
print(f"  latency --- {avg_latency:.3f} sec")
print(f"Memory consumption :  {memory_usage:.2f} MB")

mask = (mask[0, 0] > 0).detach().cpu().numpy()

# Draw resykts
plt.imshow(pil_image)
plt.imshow(mask, alpha=0.5)
x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
plt.plot(x, y, 'g-')
plt.savefig("data/benchmark_last_image.jpg")

if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, api, image_encoder, mask_decoder, time_encoder, latency, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ")
        file.write(f"nanosam-python, {args.image_encoder}, {args.mask_decoder}, {avg_encoder}, {avg_latency}, {memory_usage}\n")
        
        