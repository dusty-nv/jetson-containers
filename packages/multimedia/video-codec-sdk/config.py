
def video_codec_sdk(version, default=False):
  """
  Container that installs NVIDIA Video Codec SDK with NVENC / NVDEC (CUVID)
    https://developer.nvidia.com/nvidia-video-codec-sdk
  """
  aliases = ['nvenc', 'nvcuvid', 'cuvid']

  pkg = package.copy()

  pkg['name'] = f'video-codec-sdk:{version}'
  pkg['alias'] = [f'{x}:{version}' for x in aliases]
  pkg['build_args'] = {'NV_CODEC_VERSION': version}

  if default:
    pkg['alias'] += [pkg['name'].split(':')[0]] + aliases

  return pkg


package = [
  video_codec_sdk('12.2.72', default=True),
  video_codec_sdk('13.0.19'),
]