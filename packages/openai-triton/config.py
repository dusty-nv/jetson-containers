def openai_triton(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'openai-triton:{version}'
    
    pkg['build_args'] = {
        'OPENAITRIRON_VERSION': version,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'openai-triton:{version}-builder'

    if default:
        pkg['alias'] = 'openai-triton'
        builder['alias'] = 'openai-triton:builder'
        
    return pkg, builder

package = [
    openai_triton('2.1.0', default=True),
]

# from jetson_containers import find_container


# builder = package.copy()
# runtime = package.copy()

# builder['name'] = 'openai-triton:builder'
# builder['dockerfile'] = 'Dockerfile.builder'

# print(" ============== [openai-triton/config.py] =============== ")

# runtime['build_args'] = {
#     'BUILD_IMAGE': find_container(builder['name']),
# }

# package = [builder, runtime]
