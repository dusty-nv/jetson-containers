#!/usr/bin/env python3
print('testing tensorflow_graphics...')

import tensorflow as tf
import numpy as np
import tensorflow_graphics as tfg
print('TensorFlow version: ' + str(tf.__version__))
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
import numpy as np

print('Tensorflow graphics OK\n')