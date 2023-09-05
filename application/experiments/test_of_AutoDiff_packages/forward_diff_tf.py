import tensorflow as tf

def f(x, y):
    return x + y


if __name__ == '__main__':
    x = tf.constant(3.0)
    y = tf.constant(2.0)

    with tf.autodiff.ForwardAccumulator(
            primals=[tf.constant(100.0)],
            tangents=[x]
    ) as acc:
        z = f(x, y)
        J = acc.jvp(z)

    print(z)
    print(J)