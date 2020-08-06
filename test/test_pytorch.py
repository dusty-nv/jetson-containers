
print('testing PyTorch...')
import torch

print('PyTorch version: ' + str(torch.__version__))
print('CUDA available:  ' + str(torch.cuda.is_available()))
print('cuDNN version:   ' + str(torch.backends.cudnn.version()))

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
b = torch.randn(2, 3, 1, 4, 6)

x, lu = torch.solve(b, a)

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
print('PyTorch OK\n')
