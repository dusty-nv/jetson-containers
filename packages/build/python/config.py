
from jetson_containers import PYTHON_VERSION
from packaging.version import Version

def python(version, requires=None) -> list:
    pkg = package.copy()

    pkg['name'] = f'python:{version}'
    pkg['build_args'] = {'PYTHON_VERSION_ARG': version}

    if Version(version) == PYTHON_VERSION:
        pkg['alias'] = 'python'

    if requires:
        pkg['requires'] = requires

    return pkg

package = [
    python('3.6', '==32.*'),  # JetPack 4
    python('3.8', '<36'),     # JetPack 4 + 5
    python('3.10', '>=34'),   # JetPack 5 + 6
    python('3.11', '>=34'),   # JetPack 6
    python('3.12', '>=34'),   # JetPack 6
    python('3.13', '>=34'),   # JetPack 6
]
