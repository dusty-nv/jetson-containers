from jetson_containers import L4T_VERSION
import os

def exllama(version, branch=None, requires=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'exllama:{version}'

    if requires:
        pkg['requires'] = requires

    if default:
        pkg['alias'] = 'exllama'

    if not branch:
        branch = version

    pkg['build_args'] = {
        'EXLLAMA_VERSION': version,
        'EXLLAMA_BRANCH': branch,
        'FORCE_BUILD': os.environ.get('FORCE_BUILD', 'off'),
    }

    if L4T_VERSION.major >= 36:
        pkg['depends'] = pkg['depends'] + ['flash-attention']

    return pkg

package = [
    exllama('0.0.20', requires='>=36', default=True),
]
