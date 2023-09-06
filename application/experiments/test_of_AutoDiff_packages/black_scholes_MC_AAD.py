import tensorflow as tf


def call_payoff(x, K):
    return tf.math.maximum(x - K, 0.0)


def sim_gbm(spot, drift, vol, t, N, M, seed=None):
    dt = t[1:] - t[:-1]
    Z = tf.random.normal(shape=(M, N), mean=0.0, stddev=1.0, seed=seed, dtype=tf.float32)
    dW = tf.math.sqrt(dt) * Z
    W = tf.cumsum(tf.concat([tf.zeros(shape=(1, N)), dW], axis=0), axis=0)
    S = spot * tf.math.exp((drift - 0.5 * vol ** 2) * t + vol * W)
    return S



def call_price_MC_AAD(spot, strike, r, vol, t, N, M, seed=None):
    vars = {'spot': spot, 'strike': strike, 'r': r, 'vol': vol}
    with tf.GradientTape() as tape:
        tape.watch(vars)

        S = sim_gbm(spot, r, vol, t, N, M, seed)
        call = tf.math.reduce_mean(
            tf.exp(-t[-1] * r) * call_payoff(S[-1], strike)
        )

    return call, tape.gradient(call, vars)


if __name__ == '__main__':
    t0 = tf.constant(0.0)
    T = tf.constant(1.0)
    spot = tf.constant(100.0)
    strike = tf.constant(100.0)
    r = tf.constant(0.03)
    vol = tf.constant(0.2)
    N = tf.constant(100000)
    M = tf.constant(52 * int(T))
    seed = 1234

    t = tf.reshape(tf.linspace(t0, T, M+1), (M+1, 1))

    # Calcualte price and greeks by AAD
    price, greeks = call_price_MC_AAD(spot, strike, r, vol, t, N, M, seed)
    print('Call price =', price.numpy())
    print('Delta =', greeks['spot'].numpy())
    print('Vega =', greeks['vol'].numpy())
    print('Rho =', greeks['r'].numpy())
