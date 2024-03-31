
from jetson_containers import CUDA_ARCHITECTURES

def opencv_pip(version, requires=None, default=False):
    cv = package.copy()
    
    cv['name'] = f'opencv:{version}'
    cv['dockerfile'] = 'Dockerfile.pip'
    
    cv['build_args'] = {
        'OPENCV_VERSION': version,
        'OPENCV_PYTHON': f"{version.split('.')[0]}.x",
        'CUDA_ARCH_BIN': ','.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]),
    }
    
    if requires:
        cv['requires'] = requires
        
    if default:
        cv['alias'] = 'opencv'
        
    return cv
    
def opencv_deb(version, url, deb, requires=None, default=False):
    cv = package.copy()
    
    cv['name'] = f'opencv:{version}'
    
    cv['build_args'] = {
        'OPENCV_URL': url,
        'OPENCV_DEB': deb,
    }
    
    if requires:
        cv['requires'] = requires
        
    if default:
        cv['alias'] = 'opencv'
        
    return cv
    
package = [
    # JetPack 6
    opencv_pip('4.8.1', '==36.*', default=True),
    opencv_pip('4.9.0', '==36.*', default=False), 
    
    # JetPack 5
    opencv_pip('4.8.1', '==35.*', default=True),
    opencv_pip('4.5.0', '==35.*', default=False),
    
    # JetPack 4/5
    opencv_pip('4.5.0', '==32.*', default=True),
]

'''
package = [
    opencv_deb('4.8.1', 'https://nvidia.box.com/shared/static/ngp26xb9hb7dqbu6pbs7cs9flztmqwg0.gz', 'OpenCV-4.8.1-aarch64.tar.gz', '==36.*', default=True),
    opencv_deb('4.5.0', 'https://nvidia.box.com/shared/static/2hssa5g3v28ozvo3tc3qwxmn78yerca9.gz', 'OpenCV-4.5.0-aarch64.tar.gz', '==35.*', default=True),
    opencv_deb('4.5.0', 'https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz', 'OpenCV-4.5.0-aarch64.tar.gz', '==32.*', default=True),
]
'''