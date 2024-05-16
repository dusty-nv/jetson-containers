
from jetson_containers import CUDA_ARCHITECTURES, JETPACK_VERSION

package['build_args'] = {
    'OLLAMA_REPO': 'ollama/ollama',
    'OLLAMA_BRANCH': 'main',
    'GOLANG_VERSION': '1.22.1',
    'CMAKE_VERSION': '3.22.1',
    'JETPACK_VERSION': str(JETPACK_VERSION),
    'CMAKE_CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
}

open_webui = package.copy()

open_webui['name'] = 'ollama:open-webui'
open_webui['dockerfile'] = 'Dockerfile.open-webui'
open_webui['depends'] = ['ollama:main', 'docker']

del open_webui['alias']

package = [package, open_webui]