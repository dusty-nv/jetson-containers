
def video_codec_sdk(version, default=False):
  """
  Container that installs NVIDIA Video Codec SDK with NVENC / NVDEC (CUVID)
    https://developer.nvidia.com/nvidia-video-codec-sdk
  """
  aliases = ['nvenc', 'nvcuvid', 'cuvid']

  pkg = package.copy()
  pkg_name = pkg['name']

  pkg['name'] = f'{pkg_name}:{version}'
  pkg['alias'] = [f'{x}:{version}' for x in aliases]
  pkg['build_args'] = {'NV_CODEC_VERSION': version}

  samples = pkg.copy()

  samples['name'] = samples['name'] + '-samples'
  samples['depends'] = [pkg['name'], 'opengl', 'vulkan', 'ffmpeg', 'cmake']
  samples['build_args'] = {**samples['build_args'], 'BUILD_SAMPLES': 'on'}

  samples['alias'] = []

  if default:
    pkg['alias'] += [pkg_name] + aliases
    samples['alias'] = pkg_name + ':samples'

  return pkg, samples


package = [
  video_codec_sdk('12.2.72'),
  video_codec_sdk('13.0.19', default=True)
]
