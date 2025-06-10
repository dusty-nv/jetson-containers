import torch
import triton
import triton.language as tl

@triton.jit
def kernel(X, stride_xm, stride_xn, BLOCK: tl.constexpr) -> None:
    pass

print("testing triton...")

print('triton version: ' + str(triton.__version__))

print("testing triton kernel...")

X = torch.randn(1, device="cuda")
pgm = kernel[(1, )](X, 1, 1, BLOCK=1024)

print(pgm)

print("triton OK...")
