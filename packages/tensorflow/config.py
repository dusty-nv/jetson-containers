# config.py
# This file defines the package configurations for TensorFlow,
# using the variables imported from version.py.

from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES
from packaging.version import Version

from .version import (
    TENSORFLOW_VERSION,
    TENSORFLOW2_URL,
    TENSORFLOW2_WHL,
    TENSORFLOW1_URL,
    TENSORFLOW1_WHL,
)

def tensorflow_whl(version, whl, url, requires=None, alias=None):
    """
    Define a package for installing TensorFlow from a wheel file.
    """
    pkg = package_base.copy()
    
    # Extract the short version (e.g., '2.16') from the full version string
    short_version = Version(version.split('+')[0])  # Remove any '+nv*' suffix
    short_version = f"{short_version.major}.{short_version.minor}"
    
    pkg['name'] = f'tensorflow:{short_version}'
    pkg['alias'] = [f'tensorflow:{short_version}']
    pkg['dockerfile'] = 'Dockerfile.whl'  # Use Dockerfile.whl for installing from a wheel

    pkg['build_args'] = {
        'TENSORFLOW_WHL': whl,
        'TENSORFLOW_URL': url,
    }
    
    if requires:
        if not isinstance(requires, list):
            requires = [requires]
        pkg['requires'] = requires
    else:
        pkg['requires'] = []

    if Version(short_version) == TENSORFLOW_VERSION:
        pkg['alias'].append('tensorflow')
    
    if alias:
        pkg['alias'].append(alias)
    
    return pkg

# Initialize the base package dictionary
package_base = {
    'name': '',
    'depends': [],
    'requires': [],
    'build_args': {},
}

# Start with an empty package list
package = []

# Determine the 'requires' field based on L4T_VERSION
if L4T_VERSION.major >= 36:
    requires = '>=36'
elif L4T_VERSION.major == 35:
    requires = '==35.*'
elif L4T_VERSION.major == 34:
    requires = '==34.*'
elif L4T_VERSION.major == 32:
    requires = '==32.*'
else:
    requires = None  # Adjust as needed

# Add TensorFlow 2 package if available
if TENSORFLOW2_WHL:
    tf2_pkg = tensorflow_whl(
        str(TENSORFLOW_VERSION),
        TENSORFLOW2_WHL,
        TENSORFLOW2_URL,
        requires=requires
    )
    package.append(tf2_pkg)

# Add TensorFlow 1 package if available
if TENSORFLOW1_WHL:
    tf1_pkg = tensorflow_whl(
        '1.15.5',  # TF1 version is fixed at 1.15.5
        TENSORFLOW1_WHL,
        TENSORFLOW1_URL,
        requires=requires,
        alias='tensorflow1'
    )
    package.append(tf1_pkg)
