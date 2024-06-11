def openai_triton(version, branch=None, requires=None, default=False):
    pkg = package.copy()

    if not branch:
        branch = f'v{version}'
        
    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'openai-triton:{version}'
    pkg['alias'] = [f'triton:{version}']
    
    pkg['build_args'] = {
        'OPENAITRITON_VERSION': version,
        'OPENAITRITON_BRANCH': branch,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'openai-triton:{version}-builder'
    builder['alias'] = [f'triton:{version}-builder']
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}
    
    if default:
        pkg['alias'] += ['openai-triton', 'triton']
        builder['alias'] += ['openai-triton:builder', 'triton:builder']
        
    return pkg, builder

package = [
    openai_triton('2.1.0'),
    openai_triton('3.0.0', branch='main', default=True)
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
