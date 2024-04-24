
from jetson_containers import CUDA_ARCHITECTURES

def ctranslate2(version, requires=None, default=False):
    ct = package.copy()
    
    ct['name'] = f'ctranslate2:{version}'
    
    ct['build_args'] = {
        'CTRANSLATE_VERSION': version,
        'CTRANSLATE_BRANCH': 'v' + version if version.replace('.','').isnumeric() else version
    }
    
    if requires:
        ct['requires'] = requires

    builder = ct.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    if default:
        ct['alias'] = 'ctranslate2'
        builder['alias'] = 'ctranslate2:builder'
        
    return ct, builder
    
package = [
    ctranslate2('master', default=False),
    ctranslate2('4.2.0', default=True)
]
