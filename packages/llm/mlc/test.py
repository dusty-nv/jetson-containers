#!/usr/bin/env python3
print('testing tvm...')

import tvm
import tvm.runtime

print('tvm version:', tvm.__version__)
print('tvm cuda:', tvm.cuda().exist)

print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))

assert(tvm.cuda().exist)

print('\ntesting mlc...')

import mlc_llm
import mlc_chat

from mlc_chat import ChatModule

print(ChatModule)
