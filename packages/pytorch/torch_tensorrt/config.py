
from jetson_containers import PYTHON_VERSION

package['build_args'] = {
    'PYTHON_VERSION': PYTHON_VERSION,
    'TORCH_TRT_VERSION': 'v1.4.0',  # build setup has changed > 1.4.0, still ironing it out on aarch64
}
