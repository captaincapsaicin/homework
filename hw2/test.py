import tensorflow as tf
import numpy as np

from train_pg import build_mlp

def test_output_size():
    input_size = 3
    inputs = np.array([range(input_size)])
    x = tf.placeholder(tf.float32, [None, input_size])
    output_size = 4
    scope = 'test_scope'
    y = build_mlp(x, output_size, scope)
    # import ipdb; ipdb.set_trace()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(y, {x: inputs})
        assert output.shape == (1, output_size)

if __name__ == '__main__':
    test_output_size()
