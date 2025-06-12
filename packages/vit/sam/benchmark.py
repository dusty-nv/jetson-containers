#!/usr/bin/env python3
import os
import time
import datetime
import resource
import argparse
import socket
from urllib.parse import urlparse

import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
parser.add_argument('-i', '--images', action='append', nargs='*', help="Paths to images to test")

parser.add_argument('-r', '--runs', type=int, default=2, help="Number of inferencing runs to do (for timing)")
parser.add_argument('-w', '--warmup', type=int, default=1, help='the number of warmup iterations')

parser.add_argument('-s', '--save', type=str, default='', help='CSV file to save benchmarking results to')
   
args = parser.parse_args()

if not args.images:
    args.images = [
        "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg",
        "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg",
        "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg",
    ]
else:
    args.images = [x[0] for x in args.images]

print(args)

import requests
from tqdm import tqdm

def download_from_url(url, filename=None):

    if filename is None:
        filename = os.path.basename(urlparse(url).path)

    if not os.path.isfile(filename):

        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 # 1Kibibyte

        print(f"Downloading {filename} :")
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, download failed!")

    return os.path.abspath(filename)

def get_max_rss():  # peak memory usage in MB (max RSS - https://stackoverflow.com/a/7669482)
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  

def save_anns(cv2_image, anns):

    plt.imshow(cv2_image)

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("sam_benchmark_output.jpg")

avg_encoder=0
avg_latency=0
cv2_image=None
mask=None

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
FILENAME = os.path.basename(urlparse(args.checkpoint).path)
download_from_url(args.checkpoint, FILENAME)

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

imagepaths = []
for imageurl in args.images:
    imagepaths.append(download_from_url(imageurl))

for run in range(args.runs + args.warmup):
        
    for imagepath in imagepaths:

        cv2_image = cv2.imread(imagepath)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        time_begin=time.perf_counter()
        masks = mask_generator.generate(cv2_image)
        time_elapsed=time.perf_counter() - time_begin
        
        print(f"{imagepath}") 
        print(f"  Full pipeline :  {time_elapsed:.3f} seconds")

        if run >= args.warmup:
            avg_latency += time_elapsed
        
avg_latency /= ( args.runs * len(args.images) )

memory_usage=get_max_rss()

print(f"AVERAGE of {args.runs} runs:")
print(f"  latency --- {avg_latency:.3f} sec")
print(f"Memory consumption :  {memory_usage:.2f} MB")

save_anns(cv2_image, masks)

if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, api, checkpoint, latency, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ")
        file.write(f"sam-python, {args.checkpoint}, {avg_latency}, {memory_usage}\n")
    