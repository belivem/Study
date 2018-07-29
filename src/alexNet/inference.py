import tensorflow as tf
import numpy as np



if __name__  == "__main__":
    a = 2*np.ones([2,2,2,3])
    b = tf.nn.local_response_normalization(a,depth_radius=1,bias=0,alpha=1,beta=1)


    with tf.Session() as sess:
        print("Orinal data ==> ")
        print(a)
        print("After LRN data ==> ")
        print(sess.run(b))