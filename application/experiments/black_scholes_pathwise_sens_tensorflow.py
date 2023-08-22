import tensorflow as tf


def sim_gbm(N, t, spot, drift, vol, seed=None):
    M = tf.constant(t.shape[0])
    dt = tf.math.subtract(t[1:], t[:-1])

    Z = tf.random.normal(mean=0.0, stddev=1.0, shape=(M - 1, N), seed=seed)

    W = tf.concat([
        tf.zeros(shape=(1, N)), tf.sqrt(dt)[:, None] * Z
    ], axis=0)

    S = spot * tf.exp(((drift - 0.5 * vol ** 2) * t)[:, None] + vol * W)
    return S[-1,]


if __name__ == '__main__':
    from datetime import datetime
    seed = tf.constant(1234)
    N = tf.constant(1000)
    M = 52
    t0 = tf.constant(0.0)
    T = tf.constant(1.0)
    spot = tf.constant(100.0, dtype=tf.float32)
    drift = tf.constant(0.03, dtype=tf.float32)
    vol = tf.constant(0.2, dtype=tf.float32)
    t = tf.linspace(t0, T, M + 1)

    start = datetime.now()
    with tf.GradientTape() as tape:
        tape.watch([spot, vol])
        S = sim_gbm(N, t, spot, drift, vol, seed)

    J = tape.jacobian(S, [spot, vol])

    stop = datetime.now()
    print(stop - start)
