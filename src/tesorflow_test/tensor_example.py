
import tensorflow as tf

def test():
    a = tf.constant([1,2],dtype=tf.float32,shape=[1,2],name="constant")
    
    init = (tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(a))
    print("Hello,World!")

if __name__ == "__main__":
    test()