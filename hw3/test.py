import tensorflow as tf

def test_inverting_mask():
    done_mask_ph = tf.placeholder(tf.float32, [None])
    not_done_ph = 1 - done_mask_ph
    with tf.Session() as sess:
        value = sess.run(not_done_ph, feed_dict={done_mask_ph: [1.0]})
    assert(value == [0.0])

def test_reduce_max():
    target_q_tp1_ph = tf.placeholder(tf.float32, [None, 6])
    only_max_ph = tf.reduce_max(target_q_tp1_ph, axis=1)
    with tf.Session() as sess:
        value = sess.run(only_max_ph, feed_dict={target_q_tp1_ph: [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]})
    assert(value[0] == [[6.0]])
    assert(value[1] == [[6.0]])

if __name__ == '__main__':
    test_inverting_mask()
    test_reduce_max()
