#!/usr/bin/env python3
print('testing tensorflow_graphics...')

import tensorflow_graphics as tfg
import tensorflow as tf
import numpy as np

# Create a 2D image of a cube
image = np.zeros((128, 128, 3), dtype=np.float32)
image = tfg.image.draw_cuboid(image, [32, 32], [96, 96], [0.5, 0.5, 0.5], thickness=1)

print('image shape:', image.shape)
print('TensorFlow version: ' + str(tf.__version__))
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
print('tensorflow.graphics.__version__', tfg.__version__)

print('Tensorflow graphics OK\n')