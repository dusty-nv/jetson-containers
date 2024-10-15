from jetson_containers import CUDA_VERSION

def bitsandbytes(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'bitsandbytes:{version}'
    
    pkg['build_args'] = {
        'BITSANDBYTES_VERSION': version,
        'BITSANDBYTES_REPO': "bitsandbytes-foundation/bitsandbytes",
        'BITSANDBYTES_BRANCH': "main",
        'CUDA_INSTALLED_VERSION': int(str(CUDA_VERSION.major) + str(CUDA_VERSION.minor)),
        'CUDA_MAKE_LIB': f"cuda{str(CUDA_VERSION.major)}x"
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'bitsandbytes:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'bitsandbytes'
        builder['alias'] = 'bitsandbytes:builder'
        
    return pkg, builder

package = [
    bitsandbytes('0.44.1', default=True),
]


# from jetson_containers import CUDA_VERSION, find_container


# builder = package.copy()
# runtime = package.copy()

# builder['name'] = 'bitsandbytes:builder'
# builder['dockerfile'] = 'Dockerfile.builder'

# builder['build_args'] = {
#     'BITSANDBYTES_REPO': "dusty-nv/bitsandbytes",
#     'BITSANDBYTES_BRANCH': "main",
#     'CUDA_INSTALLED_VERSION': int(str(CUDA_VERSION.major) + str(CUDA_VERSION.minor)),
#     'CUDA_MAKE_LIB': f"cuda{str(CUDA_VERSION.major)}x"
# }

# print(" ============== [bitsandbytes/config.py] =============== ")

# runtime['build_args'] = {
#     'BUILD_IMAGE': find_container(builder['name']),
# }

# package = [builder, runtime]
