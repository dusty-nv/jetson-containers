#!/usr/bin/env python3
print("Testing multi-turboquant...")

import multi_turboquant
print(f"multi-turboquant version: {multi_turboquant.__version__}")

from multi_turboquant import get_preset
config = get_preset("balanced")
print(f"Preset 'balanced': {config}")

print('multi-turboquant OK\n')
