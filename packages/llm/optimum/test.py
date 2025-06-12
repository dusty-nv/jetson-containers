#!/usr/bin/env python3
import transformers
import onnxruntime
import optimum.version

print('optimum version:', optimum.version.__version__)
print('onnxruntime version:', onnxruntime.__version__)
print('transformers version:', transformers.__version__)
