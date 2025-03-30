#!/usr/bin/env python3
print('testing TensorFlow Text...')
import tensorflow as tf
import tensorflow_text as text

print('TensorFlow version: ' + str(tf.__version__))
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
print('tensorflow_text.__version__', text.__version__)

print(text.case_fold_utf8(['Everything not saved will be lost.']))
print(text.normalize_utf8(['Äffin']))
print(text.normalize_utf8(['Äffin'], 'nfkd'))

print("Tensorflow-Text OK\n")