
from jetson_containers import L4T_VERSION, CUDA_ARCH_LIST_INT

# latest cupy versions to support Python 3.8 (for JetPack 5)
# and Python 3.6 (for JetPack 4), respectively
if L4T_VERSION.major >= 34:
    CUPY_VERSION = 'v12.1.0'
else:
    CUPY_VERSION = 'v9.6.0'
    
# set CUPY_NVCC_GENERATE_CODE in the form of:
#   "arch=compute_53,code=sm_53;arch=compute_62,code=sm_62;arch=compute_72,code=sm_72;arch=compute_87,code=sm_87"
CUPY_NVCC_GENERATE_CODE = [f"arch=compute_{cc},code=sm_{cc}" for cc in CUDA_ARCH_LIST_INT]
CUPY_NVCC_GENERATE_CODE = ';'.join(CUPY_NVCC_GENERATE_CODE)

package['build_args'] = {
    'CUPY_VERSION': CUPY_VERSION,
    'CUPY_NVCC_GENERATE_CODE': CUPY_NVCC_GENERATE_CODE,
}
