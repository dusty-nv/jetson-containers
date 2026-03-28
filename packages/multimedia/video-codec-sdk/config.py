
def video_codec_sdk(version, nv_codec_headers_tag, default=False):
  """
  NVIDIA Video Codec SDK headers (NVENC/NVDEC) from nv-codec-headers.
  Public source, no login: https://github.com/FFmpeg/nv-codec-headers
  """
  aliases = ['nvenc', 'nvcuvid', 'cuvid']

  pkg = package.copy()
  pkg_name = pkg['name']

  pkg['name'] = f'{pkg_name}:{version}'
  pkg['alias'] = [f'{x}:{version}' for x in aliases]
  pkg['build_args'] = {
      'NV_CODEC_VERSION': version,
      'NV_CODEC_HEADERS_TAG': nv_codec_headers_tag,
  }

  if default:
    pkg['alias'] += [pkg_name] + aliases

  return pkg


package = [
  video_codec_sdk('12.2.72', nv_codec_headers_tag='n12.2.72.0'),
  video_codec_sdk('13.0.19', nv_codec_headers_tag='n13.0.19.0', default=True),
]
