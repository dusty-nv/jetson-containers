
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

print('PyTorch OK\n')
