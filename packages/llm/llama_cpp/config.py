
DEFAULT_FLAGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on -DLLAMA_CURL=on"
LEGACY_FLAGS="-DLLAMA_CUBLAS=on -DLLAMA_CUDA_F16=1"

def llama_cpp(version, branch=None, test=None, default=False, flags=DEFAULT_FLAGS):
    pkg = package.copy()

    pkg['name'] = f'llama_cpp:{version}'

    if not test:
        test = "test_model.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_S.gguf)"
        
    pkg['test'] = pkg['test'] + [test]

    if not branch:
        branch = version

    pkg['build_args'] = {
        'LLAMA_CPP_VERSION': version,
        'LLAMA_CPP_BRANCH': branch,
        'LLAMA_CPP_FLAGS': flags,
    }
    
    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    if default:
        pkg['alias'] = 'llama_cpp'
        builder['alias'] = 'llama_cpp:builder'
        
    return pkg, builder

package = [
    #llama_cpp('main', branch='main'),
    llama_cpp('0.2.57', flags=LEGACY_FLAGS),
    llama_cpp('0.2.70', flags=LEGACY_FLAGS),
    llama_cpp('0.2.83'),
    llama_cpp('0.2.90'),
    llama_cpp('0.3.1'),
    llama_cpp('0.3.2'),
    llama_cpp('0.3.5'),
    llama_cpp('0.3.6'),
    llama_cpp('0.3.7'),
    llama_cpp('0.3.8'),
    llama_cpp('0.3.9', default=True),
]
