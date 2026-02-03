
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

def build_arrow(version, branch, default=False):
    arrow = package.copy()

    arrow['name'] = f'arrow:{version}'
    arrow['build_args'] = {'ARROW_BRANCH': branch}

    if default:
        arrow['alias'] = 'arrow'

    return arrow

package = [
    build_arrow('23.0.0', 'apache-arrow-23.0.0', default=True),
]
