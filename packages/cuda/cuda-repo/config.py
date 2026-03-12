from jetson_containers import LSB_RELEASE, CUDA_VERSION, L4T_VERSION, IS_SBSA

# Define the cuda-repo package
package = {
    'name': 'cuda-repo',
    'group': 'cuda',
    'depends': ['build-essential'],
    'requires': '>=34',  # JetPack 5+
    'notes': 'Sets up NVIDIA CUDA network repositories and GPG keys',
    'build_args': {
        'DISTRO': f"ubuntu{LSB_RELEASE.replace('.','')}",
        'CUDA_VERSION_MAJOR': str(CUDA_VERSION.major),
        'L4T_VERSION_MAJOR': str(L4T_VERSION.major),
        'IS_SBSA': '1' if IS_SBSA else '0',
    }
}
