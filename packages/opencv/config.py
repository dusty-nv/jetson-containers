
from jetson_containers import CUDA_ARCHITECTURES

def opencv(version, requires=None, default=False, url=None):
    cv = package.copy()

    cv['build_args'] = {
        'OPENCV_VERSION': version,
        'OPENCV_PYTHON': f"{version.split('.')[0]}.x",
        'CUDA_ARCH_BIN': ','.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]),
    }
    
    if url:
        cv['build_args']['OPENCV_URL'] = url
        cv['name'] = f'opencv:{version}-deb'
        cv['alias'] = ['opencv:deb']
    else:
        cv['name'] = f'opencv:{version}'
        
    if requires:
        cv['requires'] = requires

    builder = cv.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    if default:
        cv['alias'] = cv.get('alias', []) + ['opencv']
        builder['alias'] = 'opencv:builder'
        
    if url:
        return cv
    else:
        return cv, builder
    
package = [
    # JetPack 6
    opencv('4.8.1', '==36.*', default=True),
    opencv('4.10.0', '==36.*', default=False), 
    
    # JetPack 5
    opencv('4.8.1', '==35.*', default=True),
    opencv('4.5.0', '==35.*', default=False),
    
    # JetPack 4
    opencv('4.5.0', '==32.*', default=True, url='https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz'),
    
    # Debians (c++)
    opencv('4.8.1', '==36.*', default=False, url='https://nvidia.box.com/shared/static/ngp26xb9hb7dqbu6pbs7cs9flztmqwg0.gz'),
    opencv('4.5.0', '==35.*', default=False, url='https://nvidia.box.com/shared/static/2hssa5g3v28ozvo3tc3qwxmn78yerca9.gz'),
]
