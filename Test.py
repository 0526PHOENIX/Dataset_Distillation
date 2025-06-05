import torch
from torch import nn, autograd

x = torch.randn(1, requires_grad=True)
y = torch.randn(1, requires_grad=True)

loss = (x - y)**2 + (y - x)**2
loss.backward()

print("x.grad:", x.grad)  # Should be non-zero
print("y.grad:", y.grad)  # Should also be non-zero