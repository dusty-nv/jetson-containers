
def vulkan_sdk(version, default=False):
  """
  Container for Vulkan SDK:

   https://github.com/KhronosGroup
   https://vulkan.lunarg.com/sdk/home

  This will pull the same sources as the prebuilt SDK,
  and rebuild them from the current architecture.
  Previous builds get cached on apt.jetson-ai-lab.io
  """
  pkg = package.copy()

  pkg['name'] = f'vulkan:{version}'

  # work with exactly version
  # if len(version.split('.')) < 4:
  #  version = version + '.0'

  pkg['build_args'] = {'VULKAN_VERSION': version}

  builder = pkg.copy()
  builder['name'] = builder['name'] + '-builder'
  builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

  if default:
    pkg['alias'] = 'vulkan'
    builder['alias'] = 'vulkan:builder'

  return pkg, builder


package = [
  vulkan_sdk('1.4.341.0', default=True),
]
