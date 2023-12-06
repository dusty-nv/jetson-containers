
from jetson_containers import CUDA_ARCHITECTURES
import copy

package['build_args'] = {
    'CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES])
}

# ggml version tracks fork
ggml = copy.deepcopy(package)

ggml['name'] = 'llama_cpp:ggml'

ggml['build_args']['LLAMA_CPP_PYTHON_REPO'] = 'dusty-nv/llama-cpp-python'
ggml['build_args']['LLAMA_CPP_PYTHON_BRANCH'] = 'v0.1.78a'

ggml['test'].extend([
    "test_model.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin)",
    "test_tokenizer.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin)"
])

# gguf version tracks main
gguf = copy.deepcopy(package)

gguf['name'] = 'llama_cpp:gguf'
ggml['alias'] = 'llama_cpp'

gguf['build_args']['LLAMA_CPP_PYTHON_REPO'] = 'abetlen/llama-cpp-python'
gguf['build_args']['LLAMA_CPP_PYTHON_BRANCH'] = 'main'

gguf['test'].extend([
    "test_model.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_S.gguf)"
])

package = [ggml, gguf]