#!/usr/bin/env python3
print('testing TensorFlow Text...')
import tensorflow_text as text

print(text.case_fold_utf8(['Everything not saved will be lost.']))
print(text.normalize_utf8(['Äffin']))
print(text.normalize_utf8(['Äffin'], 'nfkd'))

print("Tensorflow-Text OK\n")