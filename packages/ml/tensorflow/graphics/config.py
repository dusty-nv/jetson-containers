from jetson_containers import update_dependencies, PYTHON_VERSION
from packaging.version import Version
from ..ml.tensorflow.version import TENSORFLOW_VERSION

def tensorflow_graphics(version, tensorflow=None, requires=None):
    pkg = package.copy()
    
    pkg['name'] = f"tensorflow_graphics:{version.split('-')[0]}"  # remove any -rc* suffix
    
    if tensorflow:
        pkg['depends'] = update_dependencies(pkg['depends'], f"tensorflow2:{tensorflow}")
    else:
        tensorflow = TENSORFLOW_VERSION  
        
    if requires:
        pkg['requires'] = requires
   
    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {
        'TENSORFLOW_GRAPHICS_VERSION': version,
        'PYTHON_VERSION_MAJOR': PYTHON_VERSION.major,
        'PYTHON_VERSION_MINOR': PYTHON_VERSION.minor,
    }
    
    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    if not isinstance(tensorflow, Version):
        tensorflow = Version(tensorflow)

    if tensorflow == TENSORFLOW_VERSION:
        pkg['alias'] = 'tensorflow_graphics'
        builder['alias'] = 'tensorflow_graphics:builder'

    return pkg, builder
    
 
package = [
    # JetPack 5/6
    tensorflow_graphics('2.18.0', tensorflow='2.18.0', requires='==36.*'),
    tensorflow_graphics('2.19.0', tensorflow='2.19.0', requires='==36.*'),
]
