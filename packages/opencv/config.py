def install_opencv(version, url, deb, requires=None, default=False):
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
    install_opencv('4.8.1', 'https://nvidia.box.com/shared/static/ngp26xb9hb7dqbu6pbs7cs9flztmqwg0.gz', 'OpenCV-4.8.1-aarch64.tar.gz', '==36.*', default=True),
    install_opencv('4.5.0', 'https://nvidia.box.com/shared/static/2hssa5g3v28ozvo3tc3qwxmn78yerca9.gz', 'OpenCV-4.5.0-aarch64.tar.gz', '==35.*', default=True),
    install_opencv('4.5.0', 'https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz', 'OpenCV-4.5.0-aarch64.tar.gz', '==32.*', default=True),
]