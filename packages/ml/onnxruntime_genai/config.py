from jetson_containers import L4T_VERSION, CUDA_VERSION, update_dependencies
from packaging.version import Version

def onnxruntime_genai(version, branch=None, requires=None, default=False):
    ort = package.copy()

    ort['name'] = f'onnxruntime_genai:{version}'

    if requires:
        ort['requires'] = requires

    if len(version.split('.')) < 3:
        version = version + '.0'

    if not branch:
        branch = 'v' + version

    ort['build_args'] = {
        'ONNXRUNTIME_GENAI_VERSION': version,
        'ONNXRUNTIME_GENAI_BRANCH': branch,
        'CUDA_VERSION': CUDA_VERSION,
    }

    builder = ort.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if default:
        ort['alias'] = 'onnxruntime_genai'
        builder['alias'] = 'onnxruntime_genai:builder'

    return ort, builder


package = [
    onnxruntime_genai('0.11.5', requires=['>=36', '>=cu126'], default=True, branch='main')
]
