from jetson_containers import CUDA_ARCHITECTURES

def stable_diffusion_cpp(version, default=False):
    """
    Define container that builds both stable-diffusion.cpp and stable-diffusion-python.
    Different versions may have some different flag options activated.
    """
    cpp = bool(version[0] == 'b')
    pkg = package.copy()

    pkg['name'] = f'stable_diffusion_cpp:{version}'

    pkg['build_args'] = {
        'STABLE_DIFFUSION_VERSION': version[1:] if cpp else None,
        'STABLE_DIFFUSION_VERSION_PY': '0.3.17' if cpp else version,
        'STABLE_DIFFUSION_BRANCH': version if cpp else None,
        'STABLE_DIFFUSION_BRANCH_PY': 'main' if cpp else f'v{version}',
        'CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    }

    test_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    pkg['test'] = pkg['test'] + [
        f"test_model.py --model $(huggingface-downloader {test_model})"
    ]

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if default:
        pkg['alias'] = 'stable_diffusion_cpp'
        builder['alias'] = 'stable_diffusion_cpp:builder'

    return pkg, builder

package = [
    stable_diffusion_cpp('f957fa3', default=True)
]
