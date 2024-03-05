
from jetson_containers import CUDA_ARCHITECTURES, find_container


def faiss(version, tag=None, requires=None, default=False):
    builder = package.copy()
    runtime = package.copy()
    
    if default:
        builder['alias'] = 'faiss:builder'
        runtime['alias'] = 'faiss'
        
    if requires:
        builder['requires'] = requires
        runtime['requires'] = requires   
        
    if not tag:
        tag = version

    builder['name'] = f'faiss:{tag}-builder'
    runtime['name'] = f'faiss:{tag}'
    
    builder['dockerfile'] = 'Dockerfile.builder'
    
    builder['build_args'] = {
        'FAISS_VERSION': version,
        'CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    }
    
    runtime['build_args'] = {
        'BUILD_IMAGE': find_container(builder['name']),
    }
    
    return builder, runtime
    
package = [
    faiss('v1.7.3'),
    #faiss('v1.8.0'),  # encounters type_info build error sometime after be12427
    faiss('be12427', default=True),  # known good build on JP5/JP6 from 12/12/2023
]