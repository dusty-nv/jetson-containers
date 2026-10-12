from jetson_containers import CUDA_VERSION, CUDA_ARCHITECTURES
from packaging.version import Version

def opencv(version, requires=None, default=False, url=None):
    cv = package.copy()

    cv['build_args'] = {
        'OPENCV_VERSION': version,
        'OPENCV_PYTHON': f"{version.split('.')[0]}.x",
        'CUDA_ARCH_BIN': ','.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]),
    }

    if url:
        cv['build_args']['OPENCV_URL'] = url
        cv['name'] = f'opencv:{version}-deb'
        cv['alias'] = ['opencv:deb']
    else:
        cv['name'] = f'opencv:{version}'

    if requires:
        cv['requires'] = requires

    # OpenCV < 4.13 calls av_stream_get_side_data(), which FFmpeg 8.0 removed: the module
    # builds but 'import cv2' fails with an undefined symbol, so stay on FFmpeg 7.1
    if Version(version) < Version('4.13'):
        cv['depends'] = ['ffmpeg:7.1' if dep == 'ffmpeg' else dep for dep in cv['depends']]

    builder = cv.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    meta = cv.copy()
    meta['name'] = meta['name'] + '-meta'
    #meta['depends'] = [cv['name']]
    meta['dockerfile'] = 'Dockerfile.meta'

    if default:
        cv['alias'] = cv.get('alias', []) + ['opencv']
        meta['alias'] = 'opencv:meta'
        builder['alias'] = 'opencv:builder'

    if url:
        return cv
    else:
        return cv, builder, meta

package = [
    # JetPack 5/6
    opencv('4.5.0', '==35.*', default=False),
    opencv('4.8.1', '>=35', default=(CUDA_VERSION <= Version('12.2'))),
    opencv('4.10.0', '>=35', default=(CUDA_VERSION >= Version('12.4') and CUDA_VERSION < Version('12.6'))),
    opencv('4.11.0', '>=35', default=False),
    opencv('4.12.0', '>=36', default=False), # Blackwell Support
    opencv('4.13.0', '>=36', default=(CUDA_VERSION >= Version('12.6') and CUDA_VERSION < Version('13.1'))), # Thor Support
    opencv('4.14.0', '>=36', default=(CUDA_VERSION >= Version('13.2'))), # Thor Support

    # JetPack 4
    opencv('4.5.0', '==32.*', default=True, url='https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz'),

    # Debians (c++)
    opencv('4.5.0', '==35.*', default=False, url='https://nvidia.box.com/shared/static/2hssa5g3v28ozvo3tc3qwxmn78yerca9.gz'),
    opencv('4.8.1', '==36.*', default=False, url='https://nvidia.box.com/shared/static/ngp26xb9hb7dqbu6pbs7cs9flztmqwg0.gz'),
]
