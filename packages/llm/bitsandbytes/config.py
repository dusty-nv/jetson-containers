from jetson_containers import CUDA_VERSION

package['build_args'] = {
    'BITSANDBYTES_REPO': "dusty-nv/bitsandbytes",
    'BITSANDBYTES_BRANCH': "main",
    'CUDA_INSTALLED_VERSION': int(str(CUDA_VERSION.major) + str(CUDA_VERSION.minor)),
    'CUDA_MAKE_LIB': f"cuda{str(CUDA_VERSION.major)}x"
}
