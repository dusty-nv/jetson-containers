
from jetson_containers import CUDA_ARCH_LIST_INT

# set CUPY_NVCC_GENERATE_CODE in the form of:
#   "arch=compute_53,code=sm_53;arch=compute_62,code=sm_62;arch=compute_72,code=sm_72;arch=compute_87,code=sm_87"
CUPY_NVCC_GENERATE_CODE = [f"arch=compute_{cc},code=sm_{cc}" for cc in CUDA_ARCH_LIST_INT]
CUPY_NVCC_GENERATE_CODE = ';'.join(CUPY_NVCC_GENERATE_CODE)

package['build_args'] = {
    'CUPY_NVCC_GENERATE_CODE': CUPY_NVCC_GENERATE_CODE,
}

package['depends'] = ['python', 'numpy']
package['category'] = 'cuda'
