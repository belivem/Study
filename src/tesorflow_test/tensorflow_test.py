import tensorflow as tf
import math

def cross_entropy_test():
        
    #数据的label
    input_y = tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])
    #神经网络的输出   
    logits = tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])
    softmax_y = tf.nn.softmax(logits,name="softmax")

    #计算交叉熵
    corss_entropy1 = -tf.reduce_sum(input_y * tf.log(tf.clip_by_value(softmax_y,1e-10,1.0)),axis=1)
    corss_entropy2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=input_y)
    corss_entropy3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.argmax(input_y,axis=1))

    c1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=input_y)
    c2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.argmax(input_y,axis=1))

    init = (tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        print("value of c1 and c2 ==> ")
        print(sess.run(c1))
        print(sess.run(c2))
        print("softmax函数值 ==> ")
        print(sess.run(softmax_y))
        print("loss1的值 ==> ")
        print(sess.run(corss_entropy1))
        print("loss2的值 ==> ")
        print(sess.run(corss_entropy2))
        print("loss3的值 ==> ")
        print(sess.run(corss_entropy3))


#Moveing average test
def movingAverage_test():
    decay = 0.99

    v1 = tf.Variable(0,dtype=tf.float32)
    step = tf.Variable(0,dtype=tf.int32,trainable=False)

    #define moving average object
    ema = tf.train.ExponentialMovingAverage(decay,num_updates=step)
    maintain_average_op = ema.apply([v1])

    init = (tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        print("Initializer ==>")
        print(sess.run([v1,ema.average(v1)]))

        #update v1
        print("Update one ==>")
        sess.run(tf.assign(v1,5))
        sess.run(maintain_average_op)
        print(sess.run([v1,ema.average(v1)]))

        #update v1
        print("Update two ==> ")
        sess.run(tf.assign(step,10000))
        sess.run(tf.assign(v1,10))
        sess.run(maintain_average_op)
        print(sess.run([v1,ema.average(v1)]))

        #update v1
        print("Update three ==> ")
        sess.run(maintain_average_op)
        print(sess.run([v1,ema.average(v1)]))


if __name__ == "__main__":
    cross_entropy_test()