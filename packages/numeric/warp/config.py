
def warp(version, url, requires=None, default=False):
    """
    Containers for Warp (https://github.com/NVIDIA/warp)
    This defines variants for torch, jax, and all (both)
    """
    pkg = package.copy()
    name = pkg['name']

    pkg['name'] = f'{name}:{version}'

    pkg['build_args'] = {
        'WARP_VERSION': version, 
        'WARP_INSTALL': url
    }

    if requires:
        pkg['requires'] = requires

    if default:
        pkg['alias'] = ['warp']

    # add alternatives that include other libraries
    extras = ['torch', 'jax', 'all']
    packages = [pkg]

    for extra in extras:
        x = pkg.copy()
        x['name'] = f"{pkg['name']}-{extra}"
        if default:
            x['alias'] = f"{name}:{extra}"
        if extra == 'all':
            x['depends'] = x['depends'] + [y for y in extras if y != 'all'] + ['jupyterlab']
            x['test'] = 'test.sh all'
        else:
            x['depends'] = x['depends'] + [extra]
            x['test'] = f'test.sh {extra}'

        packages.append(x)

    return packages


package = [
    warp('1.7.0', 
        'https://github.com/NVIDIA/warp/releases/download/v1.7.0/warp_lang-1.7.0+cu12-py3-none-manylinux_2_34_aarch64.whl',
        requires='>=r36',
        default=True,
    ),
    warp('1.7.0', 
        'https://github.com/NVIDIA/warp/releases/download/v1.7.0/warp_lang-1.7.0+cu12-py3-none-manylinux_2_28_x86_64.whl',
        requires='x86_64',
        default=True,
    ),
]