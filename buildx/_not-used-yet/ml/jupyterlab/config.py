
from jetson_containers import LSB_RELEASE
from packaging.version import Version

def jupyterlab(version='latest', requires=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'jupyterlab:{version}'

    if version != 'latest':
      pkg['build_args'] = {
        'JUPYTERLAB_VERSION_SPEC': f'=={version}' \
        if version[0].isnumeric() else version
      }

    if requires:
      pkg['requires'] = requires

    myst = pkg.copy() # samples variant of the container

    myst['name'] = myst['name'] + '-myst'
    myst['dockerfile'] = 'Dockerfile.myst'
    myst['depends'] = [pkg['name']]

    if default:
        pkg['alias'] = 'jupyterlab'
        myst['alias'] = 'jupyterlab:myst'
        
    return pkg, myst
    
default_latest = (Version(LSB_RELEASE) >= Version('22.04'))

package = [
    jupyterlab('latest', default=default_latest),
    jupyterlab('4.2.0', default=(not default_latest))
]