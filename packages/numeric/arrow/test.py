#!/usr/bin/env python3
print('testing PyArrow...')
import pyarrow

print('PyArrow version: ' + str(pyarrow.__version__))

pyarrow.show_info()