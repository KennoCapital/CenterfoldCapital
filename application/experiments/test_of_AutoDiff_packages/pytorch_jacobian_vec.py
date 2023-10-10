import torch

x1 = torch.arange(3.0, requires_grad=True)
x2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

def f(x1, x2):
    return x1 * x2


y = f(x1, x2)

u = torch.tensor([1.0, 1.0, 1.0])
y.backward(u)  # Calculates Jacobian times u, J @ u = dy / dx @ u

print(x1.grad)
print(x2.grad)
