import torch
x = torch.rand(7, 7)
y = torch.rand(1, 7)
y = y.view(7, -1)

print(x@y)

z = torch.rand(1, 1, 1, 10)
print(z.squeeze())