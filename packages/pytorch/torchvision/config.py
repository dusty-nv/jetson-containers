from jetson_containers import L4T_VERSION, find_container


if L4T_VERSION.major >= 36:    # JetPack 6.0
    TORCHVISION_VERSION = 'v0.16.0'
elif L4T_VERSION.major >= 35:  # JetPack 5.0.2 / 5.1.x
    TORCHVISION_VERSION = 'v0.15.1'
elif L4T_VERSION.major == 34:  # JetPack 5.0 / 5.0.1
    TORCHVISION_VERSION = 'v0.12.0'
elif L4T_VERSION.major == 32:  # JetPack 4
    TORCHVISION_VERSION = 'v0.11.1'

builder = package.copy()
runtime = package.copy()

builder['name'] = 'torchvision:builder'
builder['dockerfile'] = 'Dockerfile.builder'
builder['build_args'] = {
    'TORCHVISION_VERSION': TORCHVISION_VERSION,
}

runtime['build_args'] = {
    'BUILD_IMAGE': find_container(builder['name']),
}

package = [builder, runtime]
