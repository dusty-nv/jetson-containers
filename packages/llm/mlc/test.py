#!/usr/bin/env python3
print('testing tvm...')

import tvm
import tvm.runtime

print('tvm version:', tvm.__version__)
print('tvm cuda:', tvm.cuda().exist)

print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))

assert(tvm.cuda().exist)

print('\ntesting mlc...')

# the mlc_chat module was removed around early April 2024
# and merged into the mlc_llm module with the new model builder
import mlc_llm

try:
    import mlc_chat
    from mlc_chat import ChatModule
except Exception as error:
    print(f"failed to import ChatModule from mlc_chat ({error})")
    print(f"trying to import ChatModule from mlc_llm instead...")
    from mlc_llm import ChatModule
    
print(ChatModule)
