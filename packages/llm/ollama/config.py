
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES, JETPACK_VERSION

ollama = package.copy()
ollama['name'] = 'ollama'
ollama['build_args'] = {
    'CMAKE_CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    'JETPACK_VERSION': str(JETPACK_VERSION),
    'OLLAMA_REPO': 'ollama/ollama',
    'OLLAMA_BRANCH': 'main',
    'GOLANG_VERSION': '1.22.1',
    'CMAKE_VERSION': '3.22.1',
}
ollama['test'] = 'test.sh'

package = [ollama]
