from jetson_containers import update_dependencies
from packaging.version import Version
TENSORFLOW_VERSION = '2.18.0'
def tensorflow_text(version, tensorflow2=None, requires=None):
    pkg = package.copy()
    
    pkg['name'] = f"tensorflow_text:{version.split('-')[0]}"  # remove any -rc* suffix
    
    if tensorflow2:
        pkg['depends'] = update_dependencies(pkg['depends'], f"tensorflow2:{tensorflow2}")
    else:
        tensorflow2 = TENSORFLOW_VERSION  
        
    if requires:
        pkg['requires'] = requires
   
    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {
        'TENSORFLOW_TEXT_VERSION': version,
    }
    
    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    if not isinstance(tensorflow2, Version):
        tensorflow2 = Version(tensorflow2)

    if tensorflow2 == TENSORFLOW_VERSION:
        pkg['alias'] = 'tensorflow_text'
        builder['alias'] = 'tensorflow_text:builder'

    return pkg, builder
    
 
package = [
    # JetPack 5/6
    tensorflow_text('2.18.0', tensorflow2='2.18.0', requires='==36.*'),
]
