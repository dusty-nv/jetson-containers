from jetson_containers import CUDA_VERSION, find_container


builder = package.copy()
runtime = package.copy()

builder['name'] = 'bitsandbytes:builder'
builder['dockerfile'] = 'Dockerfile.builder'

builder['build_args'] = {
    'BITSANDBYTES_REPO': "dusty-nv/bitsandbytes",
    'BITSANDBYTES_BRANCH': "main",
    'CUDA_INSTALLED_VERSION': int(str(CUDA_VERSION.major) + str(CUDA_VERSION.minor)),
    'CUDA_MAKE_LIB': f"cuda{str(CUDA_VERSION.major)}x"
}

runtime['build_args'] = {
    'BUILD_IMAGE': find_container(builder['name']),
}

package = [builder, runtime]
