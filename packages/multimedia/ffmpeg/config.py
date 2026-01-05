
def ffmpeg(source, version=None, requires=None, default=False, alias=[]):
  """
  Configure container to install FFmpeg utilities from one of these sources:

    * apt - the built-in OS version from Ubuntu apt repo
    * git - pulls/builds a version from github (or jetson-ai-lab build cache)
    * jetpack - version from NVIDIA JetPack with hw decode support for Jetson

  The 'source' argument should be string with one 'apt', 'git', 'jetpack'
  This returns a dict with the configuration to build container for ffmpeg.
  """
  pkg = package.copy()

  if requires:
    pkg['requires'] = requires

  if source == 'git':
    if not version:
      raise ValueError('ffmpeg version is required to build from git sources')
    pkg['build_args'] = {'FFMPEG_VERSION': version}
    pkg['depends'] = pkg['depends'] + ['cmake', 'video-codec-sdk']
    tag = version
  else:
    tag = source

  pkg['name'] = f'ffmpeg:{tag}'
  pkg['alias'] = alias + (['ffmpeg'] if default else [])
  pkg['build_args'] = {**pkg.get('build_args', {}), 'FFMPEG_INSTALL': source}

  return pkg


package = [
  # ffmpeg('apt', default=True),
  ffmpeg('git', version='7.1', alias=['ffmpeg:7.1'], default=False),
  ffmpeg('git', version='8.0', alias=['ffmpeg:8'], default=True),
  ffmpeg('jetpack', requires='==36.*'),
]
