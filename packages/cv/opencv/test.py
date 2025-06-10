#!/usr/bin/env python3
print('testing OpenCV...')

import cv2
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

print('OpenCV version:', str(cv2.__version__))
print(cv2.getBuildInformation())

try:
    print('\nGPU devices:', str(cv2.cuda.getCudaEnabledDeviceCount()))
except Exception as ex:
    print(ex)
    print('OpenCV was not built with CUDA')
    raise ex

# download test image with retry logic
img_url = 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg'  # More reliable URL
img_path = '/tmp/test_0.jpg'

# Configure retry strategy
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4 seconds between retries
    status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

try:
    response = http.get(img_url, allow_redirects=True, timeout=10)
    response.raise_for_status()  # Raise an exception for bad status codes
    with open(img_path, 'wb') as f:
        f.write(response.content)
    print(f'Successfully downloaded test image from {img_url}')
except Exception as e:
    print(f'Error downloading test image: {e}')
    print('Using a local test image instead...')
    # Use a simple test image creation as fallback
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(img_path, img)
    print('Created a local test image')

# load image
img_cpu = cv2.imread(img_path)
if img_cpu is None:
    raise Exception(f'Failed to load test image from {img_path}')
print(f'loaded test image from {img_path}  {img_cpu.shape}  {img_cpu.dtype}')

# test GPU processing
img_gpu = cv2.cuda_GpuMat()
img_gpu.upload(img_cpu)

img_gpu = cv2.cuda.resize(img_gpu, (int(img_cpu.shape[0]/2), int(img_cpu.shape[1]/2)))

luv = cv2.cuda.cvtColor(img_gpu, cv2.COLOR_BGR2LUV).download()
hsv = cv2.cuda.cvtColor(img_gpu, cv2.COLOR_BGR2HSV).download()
gray = cv2.cuda.cvtColor(img_gpu, cv2.COLOR_BGR2GRAY)

img_gpu = cv2.cuda.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8)).apply(gray, cv2.cuda_Stream.Null())
img_cpu = img_gpu.download()

print('OpenCV OK\n')
