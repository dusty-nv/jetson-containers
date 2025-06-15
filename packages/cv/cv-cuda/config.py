NVCV_RELEASE_URL='https://github.com/CVCUDA/CV-CUDA/releases/download/'

def cv_cuda(version, *url, default=False, requires=None):
    """ 
    Define container that installs CV-CUDA from release binaries on Github 
    
      * the main 'cv-cuda' or 'cvcuda' package is C++/CUDA/Python
      * the 'cv-cuda:cpp' or 'cvcuda:cpp' subpackages are without Python

    See here for the versions available:  https://github.com/CVCUDA/CV-CUDA
    """
    url = [x if x.startswith('http') else NVCV_RELEASE_URL + x for x in url]
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    cpp = pkg.copy()

    pkg['name'] = f'cv-cuda:{version}'
    cpp['name'] = f'cv-cuda:{version}-cpp'

    cpp['depends'] = ['cuda']
    cpp['test'] = []

    pkg['build_args'] = {
        'NVCV_VERSION': version,
        'NVCV_BINARIES': ';'.join(url)
    }

    cpp['build_args'] = {
        **pkg['build_args'], **{
            'NVCV_PYTHON': 'off',
            'NVCV_BINARIES': ';'.join(
                [x for x in url if '.whl' not in x]
            ),
        }
    }

    pkg['alias'] = [f'cvcuda:{version}']
    cpp['alias'] = [f'cvcuda:{version}-cpp']

    if default:
        pkg['alias'].extend(['cv-cuda', 'cvcuda'])
        cpp['alias'].extend(['cv-cuda:cpp', 'cvcuda:cpp'])

    return pkg, cpp


package = [
    # jetpack 6
    cv_cuda('0.14', 
        'v0.14.0-beta/v0.14_cuda_12_aarch64_Jetson_jetpack6.zip',
        default=True, requires='>=36'
    ),

    # x86
    cv_cuda('0.14', 
        'v0.14.0-beta/cvcuda-lib-0.14.0-cuda12-x86_64-linux.deb',
        'v0.14.0-beta/cvcuda-dev-0.14.0-cuda12-x86_64-linux.deb',
        'v0.14.0-beta/cvcuda-tests-0.14.0-cuda12-x86_64-linux.deb',
        'v0.14.0-beta/cvcuda_cu12-0.14.0-cp310.cp311.cp38.cp39-cp310.cp311.cp38.cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl',
        default=True, requires='x86_64'
    ),
]

