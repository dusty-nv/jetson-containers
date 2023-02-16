
print('testing PyTorch...')
import torch

print('PyTorch version: ' + str(torch.__version__))
print('CUDA available:  ' + str(torch.cuda.is_available()))
print('cuDNN version:   ' + str(torch.backends.cudnn.version()))

print(torch.__config__.show())

# quick cuda tensor test
a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))

b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))

c = a + b
print('Tensor c = ' + str(c))

# LAPACK test
print('testing LAPACK (OpenBLAS)...')

a = torch.randn(2, 3, 1, 4, 4)
b = torch.randn(2, 3, 1, 4, 4)

x, lu = torch.linalg.solve(b, a)

print('done testing LAPACK (OpenBLAS)')

# torch.nn test
print('testing torch.nn (cuDNN)...')

import torch.nn

model = torch.nn.Conv2d(3,3,3)
data = torch.zeros(1,3,10,10)
model = model.cuda()
data = data.cuda()
out = model(data)

#print(out)

print('done testing torch.nn (cuDNN)')

# CPU test (https://github.com/pytorch/pytorch/issues/47098)
print('testing CPU tensor vector operations...')

import torch.nn.functional as F
cpu_x = torch.tensor([12.345])
cpu_y = F.softmax(cpu_x)

print('Tensor cpu_x = ' + str(cpu_x))
print('Tensor softmax = ' + str(cpu_y))

if cpu_y != 1.0:
    raise ValueError('PyTorch CPU tensor vector test failed (softmax)\n')

# https://github.com/pytorch/pytorch/issues/61110
t_32 = torch.ones((3,3), dtype=torch.float32).exp()
t_64 = torch.ones((3,3), dtype=torch.float64).exp()
diff = (t_32 - t_64).abs().sum().item()

print('Tensor exp (float32) = ' + str(t_32))
print('Tensor exp (float64) = ' + str(t_64))
print('Tensor exp (diff) = ' + str(diff))

if diff > 0.1:
    raise ValueError(f'PyTorch CPU tensor vector test failed (exp, diff={diff})')
    
print('PyTorch OK\n')
