#!/usr/bin/env python3
print('testing tvm...')

import tvm
import tvm.runtime

print('tvm version:', tvm.__version__)
print('tvm cuda:', tvm.cuda().exist)

print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))

assert(tvm.cuda().exist)

print('\ntesting mlc...')

# the mlc_chat module was removed around early April 2026
# and merged into the mlc_llm module with the new model builder
import mlc_llm

try:
    print('mlc_llm version:', mlc_llm.__version__)
except Exception as error:
    print(f"failed to print mlc_llm version ({error})")

try:
    import mlc_chat                         # MLC 0.1.0
    from mlc_chat import ChatModule
    print(ChatModule)
except Exception as error:
    print(f"failed to import ChatModule from mlc_chat ({error})")
    print(f"trying to import ChatModule from mlc_llm instead...")
    try:
        from mlc_llm import ChatModule      # MLC 0.1.1
        print(ChatModule)
    except Exception as error:
        print(f"failed to import ChatModule from mlc_llm ({error})")
        print(f"trying to import MLCEngine from mlc_llm instead...")
        from mlc_llm import MLCEngine       # MLC 0.1.2+
        print(MLCEngine)
