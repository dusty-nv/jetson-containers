
def cosmos1_diffusion_renderer(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'cosmos1-diffusion-renderer:{version}'

    pkg['build_args'] = {
        'COSMOS_DIFF_RENDER_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'cosmos1-diffusion-renderer:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'cosmos1-diffusion-renderer'
        builder['alias'] = 'cosmos1-diffusion-renderer:builder'

    return pkg, builder

package = [
    cosmos1_diffusion_renderer('1.0.4', default=True),
]
