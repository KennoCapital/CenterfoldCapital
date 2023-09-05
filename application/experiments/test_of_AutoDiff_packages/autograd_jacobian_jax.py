import jax.numpy as np
from jax import jacfwd, jacrev


def f(x0, x1, x2):
    return np.array(
        [[x0 * x1], [x1 * x2], [x0 + x1 * x2]]
    ).T


@jax.jit
def g(x, y):
    return x + y


if __name__ == '__main__':
    x0 = 2.0
    x1 = 3.0
    x2 = 5.0

    #print(f(x0, x1, x2))

    J_fwd = jacfwd(f, argnums=[0, 1, 2])(x0, x1, x2)
    J_rev = jacrev(f, argnums=[0, 1, 2])(x0, x1, x2)

    #print(np.array(J_fwd))
    #print(J_rev)

    #print(type(J_fwd))

    X = np.full(shape=(3, 5), fill_value=np.nan)
    X = X.at[0, :].set(100)
    print(X)



