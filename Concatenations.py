import torch
print(torch.__version__)

x = (torch.rand(2, 3, 4) * 100).int()
print(x)
print(x.shape)

y = (torch.rand(2, 3, 4) * 100).int()
print(y)
print(y.shape)

print()


z_zero = torch.cat((x,y), 0)
#print(z_zero)
print(z_zero.shape)
print()


z_one = torch.cat((x,y), 1)
#print(z_one)
print(z_one.shape)
print()

z_two = torch.cat((x,y), 2)
print(z_two)
print(z_two.shape)

