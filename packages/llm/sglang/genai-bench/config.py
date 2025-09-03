from jetson_containers import CUDA_ARCHITECTURES, IS_SBSA

def genai_bench(version, branch=None, default=False):
    pkg = package.copy()

    if not branch:
        branch = f'v{version}'

    pkg['name'] = f'genai-bench:{version}'

    pkg['build_args'] = {
        'GENAI_BENCH_VERSION': version,
        'GENAI_BENCH_VERSION': branch,
        'IS_SBSA': IS_SBSA,
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
    }

    builder = pkg.copy()

    builder['name'] = f'genai-bench:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'genai-bench'
        builder['alias'] = 'genai-bench:builder'

    return pkg, builder

package = [
    genai_bench('0.1.0', default=True),
]
