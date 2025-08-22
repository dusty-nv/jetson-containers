#!/usr/bin/env python3
import os
import numpy as np

# Support new cuda-python layout (13.x)
try:
    from cuda.bindings import driver as cuda  # cuda-python >= 13
except Exception:
    try:
        from cuda import driver as cuda      # cuda-python ~ 12.4+
    except Exception:
        from cuda import cuda as cuda        # legacy name

try:
    from cuda.bindings import runtime as cudart  # cuda-python >= 13
except Exception:
    try:
        from cuda import runtime as cudart       # cuda-python ~ 12.4+
    except Exception:
        from cuda import cudart as cudart        # legacy name

try:
    from cuda.bindings import nvrtc as nvrtc     # cuda-python >= 13
except Exception:
    from cuda import nvrtc as nvrtc              # legacy / ~12.4+

def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
 
class KernelHelper:
    def __init__(self, code, devID):
        prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(code), b'sourceCode.cu', 0, [], []))
        CUDA_HOME = os.getenv('CUDA_HOME')
        if CUDA_HOME == None:
            raise RuntimeError('Environment variable CUDA_HOME is not defined')
        include_dirs = os.path.join(CUDA_HOME, 'include')

        # Initialize CUDA
        checkCudaErrors(cudart.cudaFree(0))

        major = checkCudaErrors(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, devID))
        minor = checkCudaErrors(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, devID))
        _, nvrtc_minor = checkCudaErrors(nvrtc.nvrtcVersion())
        use_cubin = (nvrtc_minor >= 1)
        prefix = 'sm' if use_cubin else 'compute'
        arch_arg = bytes(f'--gpu-architecture={prefix}_{major}{minor}', 'ascii')
        try:
            opts = [b'--fmad=true', arch_arg, '--include-path={}'.format(include_dirs).encode('UTF-8'),
                    b'--std=c++11', b'-default-device']
            print(code)
            print('nvcc flags:', opts)
            checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except RuntimeError as err:
            logSize = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b' ' * logSize
            checkCudaErrors(nvrtc.nvrtcGetProgramLog(prog, log))
            print(log.decode())
            print(err)
            exit(-1)

        if use_cubin:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetCUBINSize(prog))
            data = b' ' * dataSize
            checkCudaErrors(nvrtc.nvrtcGetCUBIN(prog, data))
        else:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
            data = b' ' * dataSize
            checkCudaErrors(nvrtc.nvrtcGetPTX(prog, data))

        self.module = checkCudaErrors(cuda.cuModuleLoadData(np.char.array(data)))

    def getFunction(self, name):
        return checkCudaErrors(cuda.cuModuleGetFunction(self.module, name))
 