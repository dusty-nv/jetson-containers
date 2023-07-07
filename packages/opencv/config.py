
from jetson_containers import L4T_VERSION, ARCH

if ARCH == 'aarch64':
    if L4T_VERSION.major >= 34:  # JetPack 5
        OPENCV_URL = 'https://nvidia.box.com/shared/static/2hssa5g3v28ozvo3tc3qwxmn78yerca9.gz'
        OPENCV_WHL = 'OpenCV-4.5.0-aarch64.tar.gz'
    elif L4T_VERSION.major == 32:  # JetPack 4
        OPENCV_URL = 'https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz'
        OPENCV_WHL = 'OpenCV-4.5.0-aarch64.tar.gz'
elif ARCH == 'x86_64':
    OPENCV_URL = 'https://nvidia.box.com/shared/static/vfp7krqf5bws752ts58rckpx3nyopmp1.gz'
    OPENCV_WHL = 'OpenCV-4.5.0-x86_64.tar.gz'
    
package['build_args'] = {
    'OPENCV_URL': OPENCV_URL,
    'OPENCV_WHL': OPENCV_WHL
}
