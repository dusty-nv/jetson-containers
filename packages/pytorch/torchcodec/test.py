#!/usr/bin/env python3
print('testing torchcodec...')
import torchcodec
print('torchcodec version: ' + str(torchcodec.__version__))
from torchcodec.decoders import VideoDecoder
print('torchcodec OK\n')
