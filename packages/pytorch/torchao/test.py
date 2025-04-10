#!/usr/bin/env python3
import os
import re
import time
import shutil
import requests
import argparse

print('testing torchao...')

from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer

qat_quantizer = Int8DynActInt4WeightQATQuantizer()

print('TORCHAO OK')