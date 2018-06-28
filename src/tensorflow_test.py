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


def test_confusion_matrix():

    predict = tf.constant([0,1,0,1,2,3,2,3],dtype=tf.int32,name="predict")
    labels  = tf.constant([0,1,2,0,2,1,2,3],dtype=tf.int32,name="labels")

    confusion_matrix = tf.confusion_matrix(labels,predict,num_classes=4)
    
    correct_predict = tf.diag_part(confusion_matrix)
    #compute accuracy   
    colum_sum = tf.reduce_sum(confusion_matrix,axis=0)
    accuracy = tf.truediv(correct_predict,colum_sum)
    total_accuracy = tf.reduce_mean(accuracy)

    #compute recall
    row_sum = tf.reduce_sum(confusion_matrix,axis=1)
    recall = tf.truediv(correct_predict,row_sum)
    total_recall = tf.reduce_mean(recall)

    #compute f1
    

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(predict.eval())
        print(sess.run(confusion_matrix))
        print("correct_predict ==> ")
        print(sess.run(correct_predict))
        print("colum_sum ==> ")
        print(sess.run(colum_sum))
        print("row_sum ==> ")
        print(sess.run(row_sum))
        print("accuracy ==> ")
        print(accuracy.eval())
        print(sess.run(total_accuracy))
        print("recall ==> ")
        print(recall.eval())
        print(sess.run(total_recall))


def persistence_test():
    
    ckpt_dir = "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_test/model.ckpt"
    
    v1 = tf.Variable(tf.constant(1.0,dtype=tf.float32,shape=[1]),name="v1")
    v2 = tf.Variable(tf.constant(2.0,dtype=tf.float32,shape=[1]),name="v2")

    result = tf.add(v1,v2,name="result")
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess,ckpt_dir)
        print(sess.run(result))

def restore_test():
    ckpt_dir = "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_test/model.ckpt"
    ckpt_file = "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_test/model.ckpt.meta"
    saver = tf.train.import_meta_graph(ckpt_file)
    with tf.Session() as sess:
        saver.restore(sess,ckpt_dir)


if __name__ == "__main__":
    restore_test()