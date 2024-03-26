from jetson_containers import CUDA_ARCHITECTURES, find_container

def create_packages(name, builder_dockerfile, test='test.sh'):
    builder = package.copy()
    runtime = package.copy()

    builder['name'] = f'${name}:builder'
    builder['dockerfile'] = builder_dockerfile
    builder['test'] = test
    builder['build_args'] = {
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
    }

    runtime['test'] = test
    runtime['build_args'] = {
        'BUILD_IMAGE': find_container(builder['name']),
    }

    return [builder, runtime]


exllama_v1_packages = create_packages('exllama:v1', 'Dockerfile.builder')
exllama_v2_packages = create_packages('exllama:v2', 'Dockerfile.v2.builder', test='test_v2.sh')

package = exllama_v1_packages + exllama_v2_packages
