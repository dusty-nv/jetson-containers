#!/usr/bin/env python3
print('testing tensorflow_graphics...')

import tensorflow as tf
import numpy as np
print('TensorFlow version: ' + str(tf.__version__))
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
import numpy as np
import tensorflow as tf
import trimesh

import tensorflow_graphics.geometry.transformation as tfg_transformation
from tensorflow_graphics.notebooks import threejs_visualization

# Download the mesh.
import requests

url = 'https://storage.googleapis.com/tensorflow-graphics/notebooks/index/cow.obj'
file_name = 'cow.obj'

response = requests.get(url)

# Save the file locally
with open(file_name, 'wb') as file:
    file.write(response.content)

print(f"{file_name} has been downloaded.")

# Load the mesh.
mesh = trimesh.load("cow.obj")
mesh = {"vertices": mesh.vertices, "faces": mesh.faces}

# Set the axis and angle parameters.
axis = np.array((0., 1., 0.))  # y axis.
angle = np.array((np.pi / 4.,))  # 45 degree angle.
# Rotate the mesh.
mesh["vertices"] = tfg_transformation.axis_angle.rotate(mesh["vertices"], axis,
                                                        angle).numpy()
print('Tensorflow graphics OK\n')