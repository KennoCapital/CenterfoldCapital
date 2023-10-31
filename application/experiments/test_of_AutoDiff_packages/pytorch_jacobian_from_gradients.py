import torch
from torch.func import vjp, vmap, jacrev, jacfwd
from functools import partial


def f(x0, x1, x2, x3, x4):
        return x0 ** 2 + x1 ** 3 + x2 ** 4 + x3 ** 5 + x4 ** 6


if __name__ == '__main__':
    a = torch.tensor(1.0)
    b = torch.tensor(1.0)
    c = torch.tensor(1.0)
    d = torch.tensor(1.0)
    e = torch.tensor(1.0)

    x0_values = torch.arange(1.0, 10000.0, 1.0)

    # MANUAL "SLOW" IMPLEMENTATION
    g = torch.autograd.functional.jacobian(func=f, inputs=(a, b, c, d, e))
    print(g)
    """ (tensor(2.), tensor(3.), tensor(4.), tensor(5.), tensor(6.)) """

    g_list = []
    for x0 in x0_values:
        g = torch.autograd.functional.jacobian(func=f, inputs=(x0, b, c, d, e))
        g_list.append(torch.hstack(g))

    J = torch.vstack(g_list)
    print(J)

    """
    tensor([[2., 3., 4., 5., 6.],
            [4., 3., 4., 5., 6.],
            [6., 3., 4., 5., 6.],
            [8., 3., 4., 5., 6.],
            [10., 3., 4., 5., 6.],
            [12., 3., 4., 5., 6.],
            [14., 3., 4., 5., 6.],
            [16., 3., 4., 5., 6.],
            [18., 3., 4., 5., 6.]])
    """

    # MANUAL "FAST" IMPLEMENTATION
    f_res, vjp_f = vjp(partial(f, x1=b, x2=c, x3=d, x4=e), a)
    Jv = vmap(vjp_f)(x0_values)
    print(Jv)

    # AUTOMATIC IMPLEMENTATION (eats all your memory when used on large vectors)
    Jv2 = jacrev(f, argnums=0)(x0_values, b, c, d, e) @ torch.ones_like(x0_values)
    Jv3 = jacfwd(f, argnums=0)(x0_values, b, c, d, e) @ torch.ones_like(x0_values)

