from jetson_containers import CUDA_ARCHITECTURES, find_container

builder = package.copy()
runtime = package.copy()

builder['name'] = 'auto_awq:builder'
builder['dockerfile'] = 'Dockerfile.builder'

builder['build_args'] = {
    'AUTOAWQ_BRANCH': 'main',
    'AUTOAWQ_CUDA_ARCH': ','.join([str(x) for x in CUDA_ARCHITECTURES])
}

runtime['build_args'] = {
    'BUILD_IMAGE': find_container(builder['name']),
}

package = [builder, runtime]
