#!/bin/bash
python3 - <<EOF
print('testing KAOLIN...')
import copy
import glob
import math
import logging
import numpy as np
import os
import sys
import torch

try:
    import matplotlib.pyplot as plt
except Exception as e:
    pass


FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

# Folder with common data samples (<kaolin_root>/sample_data)
COMMON_DATA_DIR = os.path.realpath(os.path.join(FILE_DIR, os.pardir, os.pardir, 'sample_data'))

# Folder with data specific to examples (<kaolin_root>/examples/samples)
EXAMPLES_DATA_DIR = os.path.realpath(os.path.join(FILE_DIR, os.pardir, 'samples'))

import kaolin as kal

def print_tensor(t, *args, **kwargs):
    print(kal.utils.testing.tensor_info(t, *args, **kwargs))

# Let's construct a simple unbatched mesh
V, F, Fsz = 10, 5, 3
mesh = kal.rep.SurfaceMesh(vertices=torch.rand((V, 3)).float(),
                           faces=torch.randint(0, V, (F, Fsz)).long(),
                           allow_auto_compute=False)  # disable auto-compute for now
print_tensor(mesh.vertices, name='vertices')
print_tensor(mesh.faces, name='faces')
print_tensor(mesh.face_vertices, name='face_vertices')

# Now let's enable auto-compute
mesh.allow_auto_compute=True
print_tensor(mesh.face_vertices, name='face_vertices (auto-computed)')

print('KAOLIN OK\\n')
EOF
