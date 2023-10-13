import torch
from torch.autograd.functional import jacobian


def mySqProd(x, Z):
    return x ** 2 * Z


x = torch.arange(5.0, requires_grad=True)
Z = torch.randn(size=(5, ), generator=torch.Generator().manual_seed(1234))

J = jacobian(func=mySqProd, inputs=(x, Z), vectorize=True, strategy='forward-mode')
print(J, '\n')

J = jacobian(func=mySqProd, inputs=(x, Z), strategy='reverse-mode', create_graph=True)
print(J, '\n')

y = mySqProd(x, Z)
y.backward(gradient=torch.ones_like(x))
print(x.grad)
print(Z * 2 * x)

