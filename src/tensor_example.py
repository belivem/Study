
import tensorflow as tf
MODEL_PATH = "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_test/test_model.ckpt"


def test():

    v1 = tf.Variable(tf.truncated_normal([3,2,3],dtype=tf.float32),name="v1")

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        print(v1.get_shape()[1].value)

if __name__ == "__main__":
    test()