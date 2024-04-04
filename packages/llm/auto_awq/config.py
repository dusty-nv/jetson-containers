from jetson_containers import CUDA_ARCHITECTURES

def AutoAWQ(version, kernels_version, default=False):
    pkg = package.copy()

    pkg['name'] = f'auto_awq:{version}'

    if default:
        pkg['alias'] = 'auto_awq'
    
    pkg['build_args'] = {
        'AUTOAWQ_VERSION': version,
        'AUTOAWQ_KERNELS_VERSION': kernels_version,
        'AUTOAWQ_CUDA_ARCH': ','.join([str(x) for x in CUDA_ARCHITECTURES])
    }
    
    return pkg

package = [
    AutoAWQ('0.2.4', '0.0.6', default=True),
]
