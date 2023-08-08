
from jetson_containers import CUDA_ARCHITECTURES

# AWQ package is for Orin only:
#  ptxas /tmp/tmpxft_000000b4_00000000-7_gemm_cuda_gen.compute_72.ptx, line 889; error
#   Modifier '.m8n8' requires .target sm_75 or higher
#   Feature '.m16n8k16' requires .target sm_80 or higher
package['build_args'] = {
    'TORCH_CUDA_ARCH_LIST': '8.7',
}
