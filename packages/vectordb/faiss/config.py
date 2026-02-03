
def faiss(version, branch=None, requires=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'faiss:{version}'

    if len(version.split('.')) < 3:
        version = version + '.0'

    if not branch:
        branch = 'v' + version

    pkg['build_args'] = {
        'FAISS_VERSION': version,
        'FAISS_BRANCH': branch,
    }

    if requires:
        pkg['requires'] = requires

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if default:
        pkg['alias'] = 'faiss'
        builder['alias'] = 'faiss:builder'

    return pkg, builder

package = [
    faiss('1.13.3', default=True),
    #faiss('be12427', default=True),  # known good build on JP5/JP6 from 12/12/2023
]
