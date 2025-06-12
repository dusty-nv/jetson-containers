#!/usr/bin/env python3
import os
import re
import time
import shutil
import requests
import argparse

print('testing PYTORCH3D...')
import os
import sys
import torch
need_pytorch3d=False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d=True
print('PYTORCH3D OK')