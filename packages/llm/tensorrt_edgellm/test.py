#!/usr/bin/env python3
print('testing tensorrt_edgellm...')

import tensorrt_edgellm
print('tensorrt_edgellm version:', tensorrt_edgellm.__version__)

import subprocess
import os

source_dir = os.environ.get('TENSORRT_EDGELLM_DIR', '/opt/TensorRT-Edge-LLM')
build_dir = os.path.join(source_dir, 'build')

llm_build = os.path.join(build_dir, 'examples', 'llm', 'llm_build')
llm_inference = os.path.join(build_dir, 'examples', 'llm', 'llm_inference')

if os.path.isfile(llm_build):
    result = subprocess.run([llm_build, '--help'], capture_output=True, text=True)
    print('llm_build --help:', 'OK' if result.returncode == 0 else 'FAILED')
else:
    print(f'llm_build not found at {llm_build} (C++ runtime may not be built)')

if os.path.isfile(llm_inference):
    result = subprocess.run([llm_inference, '--help'], capture_output=True, text=True)
    print('llm_inference --help:', 'OK' if result.returncode == 0 else 'FAILED')
else:
    print(f'llm_inference not found at {llm_inference} (C++ runtime may not be built)')

print('\ntensorrt_edgellm OK\n')
