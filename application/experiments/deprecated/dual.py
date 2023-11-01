import numpy as np

class Dual:
    """
        Class implementing dual numbers: https://en.wikipedia.org/wiki/Dual_number

        To use in the context of Automatic Differentiation for to track `x` through the function `f` do
            y = f(Dual(x, 1)).

        https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers
    """
    def __init__(self, real, dual=0.0):
        self.real = real
        self.dual = dual

    def __pos__(self):
        return self

    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def __add__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.real + other.real, self.dual + other.dual)

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return -self + other

    def __mul__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.real / other.real, (self.dual * other.real - self.real * other.dual) / other.real ** 2)

    def __rtruediv__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(other.real / self.real, (other.dual * self.real - other.real * self.dual) / self.real ** 2)

    def __ge__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return self.real >= other.real

    def __gt__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return self.real > other.real

    def __le__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return self.real <= other.real

    def __lt__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return self.real < other.real

    def __pow__(self, power):
        return Dual(self.real ** power, self.dual * power * self.real ** (power-1))

    def __abs__(self):
        return Dual(np.abs(self.real), self.dual * np.sign(self.real))

    def __str__(self):
        return '(%.4f, %.4f)' % (self.real, self.dual)

    def __repr__(self):
        if self.dual >= 0:
            return f'{self.real} + {self.dual}ε'
        else:
            return f'{self.real} - {-self.dual}ε'

"""
    Dual safe functions
"""

def as_dual(x):
    if isinstance(x, Dual):
        return x
    else:
        return Dual(x)


def exp(x):
    if isinstance(x, Dual):
        return Dual(np.exp(x.real), x.dual * np.exp(x.real))
    else:
        return np.exp(x)


def log(x):
    if isinstance(x, Dual):
        if x.real > 0.0:
            return Dual(np.log(x.real), x.dual / x.real)
        else:
            raise ValueError
    else:
        return np.log(x)


def sqrt(x):
    if isinstance(x, Dual):
        if x.real > 0.0:
            return Dual(np.sqrt(x.real), x.dual * 0.5 * x.real**(-0.5))
        else:
            raise ZeroDivisionError
    else:
        return np.sqrt(x)


def erf_deriv(x):
    """Derivative of the error function"""
    return 2 / np.sqrt(np.pi) * np.exp(-x ** 2)


def erf(x):
    """Numerical approximation of the error function: https://en.wikipedia.org/wiki/Error_function#Numerical_approximations"""
    if isinstance(x, Dual):
        a = x.real
        b = x.dual
    else:
        a = x

    if a < 0.0:
        return - erf(-x)

    c1 = 0.0705230784
    c2 = 0.0422820123
    c3 = 0.0092705272
    c4 = 0.0001520143
    c5 = 0.0002765672
    c6 = 0.0000430638

    res = 1 - 1 / (1 + c1*a + c2*a**2 + c3*a**3 + c4*a**4 + c5*a**5 + c6*a**6) ** 16

    if isinstance(x, Dual):
        return Dual(res, b * erf_deriv(a))

    return res


def normal_pdf(x, mu=0.0, sigma=1.0):
    return 1 / np.sqrt(2 * sigma * np.pi) * exp(-0.5 * ((x-mu) / sigma) ** 2)


def normal_cdf(x, mu=0.0, sigma=1.0):
    inner = (x-mu) / (sigma * sqrt(2))
    return 0.5 * (1 + erf(inner))


""" TEST """
if __name__ == '__main__':
    x = Dual(0, 1)
    y = Dual(2, 1)
    print(x)
    print(y)

    print('')

    print('x + y =', repr(x + y))
    print('x - y =', repr(x - y))
    print('x * y =', repr(x * y))
    print('x / y =', repr(x / y))

    print('')

    print(y + 2.0, 2.0 + y)
    print(y - 2.0, 2.0 - y)
    print(y * 2.0, 2.0 * y)
    print(y / 2.0, 2.0 / y)

    print(y.real / 2, (y.dual * 2 - y.real * 1) / 2**2)
    print(2 / y.real, (1 * y.real - 2 * y.dual) / y.real**2)

    print('')

    print('exp(x) =', repr(exp(x)))     # exp'(x) = exp(x)
    print('log(y) =', repr(log(y)))     # log'(x) = 1 / x
    print('sqrt(y) =', repr(sqrt(y)))   # sqrt'(x) = 0.5 * x ** -0.5
    print('y ** 3 =', repr(y ** 3))

    print('')

    print('erf(x) =', erf(x), 2 / np.sqrt(np.pi) * np.exp(-x.real**2))
    print('N\'(x) =', normal_pdf(x))
    print('N(x) = ', normal_cdf(x))

    """ BLACK SCHOLES """

    S = Dual(100, 1)
    K = 100
    r = 0.03
    sigma = 0.2
    t = 0.0
    T = 1.0
    option_type = 'CALL'

    def black_scholes_formula(S, K, r, sigma, t, T, option_type):
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * sqrt(T - t))
        d2 = d1 - (sigma * sqrt(T - t))

        if option_type == 'CALL':
            return normal_cdf(d1) * S - normal_cdf(d2) * K * np.exp(-r*(T-t))
        elif option_type == 'PUT':
            return normal_cdf(-d2) * K * np.exp(-r*(T-t)) - normal_cdf(-d1) * S
        else:
            raise NotImplementedError

    bs = black_scholes_formula(S=S, K=K, r=r, sigma=sigma, t=t, T=T, option_type=option_type)

    print(bs)
