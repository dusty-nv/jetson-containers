
import copy
from jetson_containers import find_container

def create_packages(name, repo, branch, test=None, default=False):
    builder = copy.deepcopy(package)
    runtime = copy.deepcopy(package)

    builder['name'] = f'${name}:builder'
    builder['dockerfile'] = 'Dockerfile.builder'

    if test is not None:
        builder['test'].extend(test)
        runtime['test'].extend(test)

    builder['build_args'] = {
        'LLAMA_CPP_PYTHON_REPO': repo,
        'LLAMA_CPP_PYTHON_BRANCH': branch,
    }

    runtime['build_args'] = {
        'BUILD_IMAGE': find_container(builder['name']),
    }

    if default:
        runtime['alias'] = 'llama_cpp'

    return [builder, runtime]

ggml_packages = create_packages(
    'llama_cpp:ggml',
    'dusty-nv/llama-cpp-python',
    'v0.1.78a',
    test=[
        "test_model.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin)",
        "test_tokenizer.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin)"
    ],
)

gguf_packages = create_packages(
    'llama_cpp:gguf',
    'abetlen/llama-cpp-python',
    'main',
    test=['test_model.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_S.gguf)'],
    default=True
)

package = [ggml_packages, gguf_packages]
