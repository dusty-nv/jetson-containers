
from packaging.version import Version

def onnxruntime(version, branch=None, requires=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'onnxruntime:{version}'

    if default:
        pkg['alias'] = 'onnxruntime'
    
    if requires:
        pkg['requires'] = requires
        
    if len(version.split('.')) < 3:
        version = version + '.0'
            
    if not branch:
        branch = 'v' + version
    
    pkg['build_args'] = {
        'ONNXRUNTIME_VERSION': version,
        'ONNXRUNTIME_BRANCH': branch,
        'ONNXRUNTIME_FLAGS': '', 
    }
    
    if Version(version) >= Version('1.13'):
        pkg['build_args']['ONNXRUNTIME_FLAGS'] = '--allow_running_as_root'
    
    return pkg
    
    
package = [
    onnxruntime('1.17', requires='>=35', default=True),
    onnxruntime('1.11', requires='==32.*', default=True),
]