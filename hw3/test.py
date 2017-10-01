import tensorflow as tf

def test_inverting_mask():
    done_mask_ph          = tf.placeholder(tf.float32, [None])
    not_done_ph = 1 - done_mask_ph
    with tf.Session() as sess:
        value = sess.run(not_done_ph, feed_dict={done_mask_ph: [1.0]})
    assert(value == [0.0])


if __name__ == '__main__':
    test_inverting_mask()
