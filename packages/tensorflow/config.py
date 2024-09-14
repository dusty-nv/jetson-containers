from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES
from packaging.version import Version

from .version import (
    TENSORFLOW_VERSION,
    TENSORFLOW2_URL,
    TENSORFLOW2_WHL,
    TENSORFLOW1_URL,
    TENSORFLOW1_WHL,
)

def tensorflow_pip(version, requires=None, alias=None):
    pkg = package_base.copy()

    short_version = Version(version.split('-')[0]) 
    short_version = f"{short_version.major}.{short_version.minor}"

    pkg['name'] = f'tensorflow:{short_version}'
    pkg['dockerfile'] = 'Dockerfile.pip'

    pkg['build_args'] = {
        'TENSORFLOW_VERSION': version,
    }

    if requires:
        if not isinstance(requires, list):
            requires = [requires]
        pkg['requires'] = requires

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder.get('build_args', {}), 'FORCE_BUILD': 'on'}

    pkg['alias'] = [f'tensorflow:{short_version}']
    builder['alias'] = [f'tensorflow:{short_version}-builder']

    if Version(short_version) == TENSORFLOW_VERSION:
        pkg['alias'].append('tensorflow')
        builder['alias'].append('tensorflow:builder')

    if alias:
        pkg['alias'].append(alias)

    return pkg, builder

def tensorflow_whl(version, whl, url, requires=None, alias=None):
    pkg = package_base.copy()

    short_version = Version(version.split('+')[0])  
    short_version = f"{short_version.major}.{short_version.minor}"

    pkg['name'] = f'tensorflow:{short_version}'
    pkg['dockerfile'] = 'Dockerfile.whl'

    pkg['build_args'] = {
        'TENSORFLOW_WHL': whl,
        'TENSORFLOW_URL': url,
    }

    if requires:
        if not isinstance(requires, list):
            requires = [requires]
        pkg['requires'] = requires

    pkg['alias'] = [f'tensorflow:{short_version}']

    if Version(short_version) == TENSORFLOW_VERSION:
        pkg['alias'].append('tensorflow')

    if alias:
        pkg['alias'].append(alias)

    return pkg

package_base = {
    'name': '',
    'depends': [],
    'requires': [],
    'build_args': {},
}

package = []

if L4T_VERSION.major >= 36:
    requires = '>=36'
else:
    requires = f'=={L4T_VERSION.major}.*'

if TENSORFLOW2_WHL and TENSORFLOW2_URL:
    tf_whl_pkg = tensorflow_whl(
        str(TENSORFLOW_VERSION),
        TENSORFLOW2_WHL,
        TENSORFLOW2_URL,
        requires=requires
    )
    package.append(tf_whl_pkg)

tf_pip_pkg, tf_pip_builder = tensorflow_pip(
    str(TENSORFLOW_VERSION),
    requires=requires
)
package.extend([tf_pip_pkg, tf_pip_builder])

if TENSORFLOW1_WHL and TENSORFLOW1_URL:
    tf1_whl_pkg = tensorflow_whl(
        '1.15.5',
        TENSORFLOW1_WHL,
        TENSORFLOW1_URL,
        requires=requires,
        alias='tensorflow1'
    )
    package.append(tf1_whl_pkg)