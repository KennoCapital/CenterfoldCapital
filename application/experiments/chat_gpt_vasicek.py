import numpy as np

# Parameters (same as before)
a = 0.86
b = 0.09
sigma = 0.0148
r0 = 0.08
t = 0.0
delta_t = 0.25  # Time interval
n = 4  # Number of maturities
T = [0.25, 0.50, 0.75, 1.0]  # Maturities

def B(t):
    return (1 - np.exp(a * t)) / a

def A(t):
    (self.b - self.sigma ** 2 / (2 * self.a ** 2)) * (t - B) - \
    self.sigma ** 2 * B ** 2 / (4 * self.a)