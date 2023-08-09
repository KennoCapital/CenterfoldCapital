import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
N = lambda x: tfd.Normal(loc=0.0, scale=1.0).cdf(x)


@tf.function
def black_scholes(S, K, T, t, r, sigma):
    d1 = (tf.math.log(S / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * tf.math.sqrt(T - t))
    d2 = d1 - sigma * tf.math.sqrt(T - t)
    return S * N(d1) - K * tf.math.exp(-r * (T-t)) * N(d2)


def black_scholes_AAD(S, K, T, t, r, sigma):
    keys = black_scholes_AAD.__code__.co_varnames[:black_scholes_AAD.__code__.co_argcount]
    values = [S, K, T, t, r, sigma]
    vars = dict(zip(keys, values))

    with tf.GradientTape() as tape:
        tape.watch(vars)
        price = black_scholes(**vars)

    res = {'price': price}
    res.update(tape.gradient(price, vars))
    return res


if __name__ == "__main__":
    S = tf.constant(100.0)
    K = tf.constant(100.0)
    T = tf.constant(1.0)
    t = tf.constant(0.0)
    r = tf.constant(0.03)
    sigma = tf.constant(0.2)

    # Calculate the Black Scholes price and greeks
    calc = black_scholes_AAD(S, K, T, t, r, sigma)

    for k, v in calc.items():
        print(k, v.numpy())

