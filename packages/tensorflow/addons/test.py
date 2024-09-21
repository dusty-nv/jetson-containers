#!/usr/bin/env python3
print('testing tensorflow_addons...')

import tensorflow as tf
import tensorflow_addons as tfa

print('TensorFlow version: ' + str(tf.__version__))
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
print('tensorflow_addons.__version__', tfa.__version__)

print('Tensorflow addons OK\n')