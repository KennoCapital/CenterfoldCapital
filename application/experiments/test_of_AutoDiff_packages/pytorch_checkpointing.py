import torch
from torch.autograd.functional import jacobian
from torch.autograd.graph import

x1 = torch.tensor(2.0)
x2 = torch.tensor(3.0)


def y(a, b):
    return a * b

def z(a, b):
    return a ** 2 + b ** 2

def comp(a, b):
    return y(a, b), z(a, b)


print(comp(x1, x2))


J = jacobian(func=comp, inputs=(x1, x2), create_graph=True)
print(J)