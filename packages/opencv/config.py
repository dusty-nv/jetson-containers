
from jetson_containers import CUDA_ARCHITECTURES

def opencv(version, requires=None, default=False, url=None):
    cv = package.copy()
    
    cv['name'] = f'opencv:{version}'
    
    cv['build_args'] = {
        'OPENCV_VERSION': version,
        'OPENCV_PYTHON': f"{version.split('.')[0]}.x",
        'CUDA_ARCH_BIN': ','.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]),
    }
    
    if url:
        cv['build_args']['OPENCV_URL'] = url
        
    if requires:
        cv['requires'] = requires

    builder = cv.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    if default:
        cv['alias'] = 'opencv'
        builder['alias'] = 'opencv:builder'
        
    return cv, builder
    
package = [
    # JetPack 6
    opencv('4.8.1', '==36.*', default=True),
    opencv('4.9.0', '==36.*', default=False), 
    
    # JetPack 5
    opencv('4.8.1', '==35.*', default=True),
    opencv('4.5.0', '==35.*', default=False),
    
    # JetPack 4/5
    opencv('4.5.0', '==32.*', default=True, url='https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz'),
]
