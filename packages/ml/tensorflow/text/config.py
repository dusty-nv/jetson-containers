from jetson_containers import update_dependencies, PYTHON_VERSION
from packaging.version import Version
from ..ml.tensorflow.version import TENSORFLOW_VERSION

def tensorflow_text(version, tensorflow=None, requires=None):
    pkg = package.copy()

    pkg['name'] = f"tensorflow_text:{version.split('-')[0]}"  # remove any -rc* suffix

    if tensorflow:
        pkg['depends'] = update_dependencies(pkg['depends'], f"tensorflow2:{tensorflow}")
    else:
        tensorflow = TENSORFLOW_VERSION

    if requires:
        pkg['requires'] = requires

    if len(version.split('.')) < 3:
        version = version + '.0'

    pkg['build_args'] = {
        'TENSORFLOW_TEXT_VERSION': version,
        'PYTHON_VERSION_MAJOR': PYTHON_VERSION.major,
        'PYTHON_VERSION_MINOR': PYTHON_VERSION.minor,
    }

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if not isinstance(tensorflow, Version):
        tensorflow = Version(tensorflow)

    if tensorflow == TENSORFLOW_VERSION:
        pkg['alias'] = 'tensorflow_text'
        builder['alias'] = 'tensorflow_text:builder'

    return pkg, builder


package = [
    # JetPack 5/6
    tensorflow_text('2.21.0', tensorflow='2.21.0', requires='>=36'),
]
