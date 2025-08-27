#!/usr/bin/env python3
import tvm
import tvm.runtime

print('tvm version:', tvm.__version__)
print('tvm cuda:', tvm.cuda().exist)

print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))

assert(tvm.cuda().exist)
print('tvm OK')