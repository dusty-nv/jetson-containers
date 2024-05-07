from jetson_containers import find_container


builder = package.copy()
runtime = package.copy()

builder['name'] = 'openai-triton:builder'
builder['dockerfile'] = 'Dockerfile.builder'

runtime['build_args'] = {
    'BUILD_IMAGE': find_container(builder['name']),
}

package = [builder, runtime]
