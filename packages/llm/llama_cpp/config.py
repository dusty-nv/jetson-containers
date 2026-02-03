GGUF_FLAGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on -DLLAMA_CURL=on -DGGML_CUDA_FA_ALL_QUANTS=ON"
GGML_FLAGS="-DLLAMA_CUBLAS=on -DLLAMA_CUDA_F16=1"
from jetson_containers import CUDA_ARCHITECTURES

def llama_cpp(version, default=False, flags=GGUF_FLAGS):
    """
    Define container that builds both llama.cpp and llama-cpp-python.
    Different versions may have some different flag options activated.
    """
    cpp = bool(version[0] == 'b')
    pkg = package.copy()

    pkg['name'] = f'llama_cpp:{version}'

    pkg['build_args'] = {
        'LLAMA_CPP_VERSION': version[1:] if cpp else None,
        'LLAMA_CPP_VERSION_PY': '0.3.16' if cpp else version,
        'LLAMA_CPP_BRANCH': version if cpp else None,
        'LLAMA_CPP_BRANCH_PY': 'main' if cpp else f'v{version}',
        'LLAMA_CPP_FLAGS': flags,
        'CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    }

    if cpp:
        test_model = "bartowski/Qwen_Qwen3-1.7B-GGUF/Qwen_Qwen3-1.7B-Q4_K_M.gguf"
    else:
        test_model = "TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_S.gguf"

    pkg['test'] = pkg['test'] + [
        f"test_model.py --model $(huggingface-downloader {test_model})"
    ]

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if default:
        pkg['alias'] = 'llama_cpp'
        builder['alias'] = 'llama_cpp:builder'

    return pkg, builder

package = [
    llama_cpp('0.2.57', flags=GGML_FLAGS),
    llama_cpp('0.2.70', flags=GGML_FLAGS),
    # llama_cpp_python appears abandoned (4/25)
    # so we changed over to llama.cpp branches
    llama_cpp('b5255'),
    llama_cpp('b7917', default=True)
]
