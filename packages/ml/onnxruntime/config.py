
from packaging.version import Version

def onnxruntime(version, branch=None, requires=None, default=False):
    ort = package.copy()

    ort['name'] = f'onnxruntime:{version}'

    if requires:
        ort['requires'] = requires
        
    if len(version.split('.')) < 3:
        version = version + '.0'
            
    if not branch:
        branch = 'v' + version
    
    ort['build_args'] = {
        'ONNXRUNTIME_VERSION': version,
        'ONNXRUNTIME_BRANCH': branch,
        'ONNXRUNTIME_FLAGS': '', 
    }
    
    if Version(version) >= Version('1.13'):
        ort['build_args']['ONNXRUNTIME_FLAGS'] = '--allow_running_as_root'
    
    builder = ort.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    if default:
        ort['alias'] = 'onnxruntime'
        builder['alias'] = 'onnxruntime:builder'
    
    return ort, builder
    
    
package = [
    onnxruntime('1.22', requires=['>=36', '>=cu128'], default=False, branch='main'),
    onnxruntime('1.21', requires=['>=36', '>=cu124'], default=True),
    onnxruntime('1.20.1', requires=['>=36', '>=cu124'], default=False),
    onnxruntime('1.20', requires=['>=36', '>=cu124'], default=False),
    onnxruntime('1.19.2', requires=['>=36', '>=cu124'], default=False),
    onnxruntime('1.17', requires=['>=36', '<=cu122'], default=True),
    onnxruntime('1.16.3', requires='==35.*', default=True),
    onnxruntime('1.11', requires='==32.*', default=True),
]