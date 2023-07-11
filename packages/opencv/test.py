#!/usr/bin/env python3
print('testing OpenCV...')

import cv2
import sys
import wget

print('OpenCV version:', str(cv2.__version__))
print(cv2.getBuildInformation())

try:
    print('\nGPU devices:', str(cv2.cuda.getCudaEnabledDeviceCount()))
except Exception as ex:
    print(ex)
    print('OpenCV was not built with CUDA')
    sys.exit()

# download test image    
img_url = 'https://raw.githubusercontent.com/dusty-nv/jetson-containers/59f840abbb99f22914a7b2471da829b3dd56122e/test/data/test_0.jpg'
img_path = '/tmp/test_0.jpg'

wget.download(img_url, img_path)

# load image
img_cpu = cv2.imread(img_path)
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
