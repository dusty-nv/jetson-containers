
print('testing CuPy...')
import cupy as cp

print('CuPy version: ' + str(cp.__version__))
print(cp.show_config())

print('running CuPy GPU array test...')

x_gpu = cp.array([1, 2, 3])
print(x_gpu)
l2_gpu = cp.linalg.norm(x_gpu)
print(l2_gpu)

print('done CuPy GPU array test')
print('CuPy OK\n')
