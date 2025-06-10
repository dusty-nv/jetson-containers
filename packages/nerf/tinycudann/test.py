#!/usr/bin/env python3
print('testing tinycudann...')

import torch
import tinycudann as tcnn

net = tcnn.Network(
	n_input_dims=3,
	n_output_dims=3,
	network_config={
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 16,
		"n_hidden_layers": 2,
	},
).cuda()

x = torch.rand(256, 3, device='cuda')
y = net(x)
y.sum().backward() # OK


x2 = torch.rand(256, 3, device='cuda')
y = net(x)
y2 = net(x2)
(y + y2).sum().backward() # RuntimeError: Must call forward() before calling backward()

print("success!")


print('tcnn OK\n')

