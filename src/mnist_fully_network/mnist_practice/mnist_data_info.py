import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_path = "/Users/liyanan/Documents/Test/Tensorflow/data/mnist_data/"

#print mnist data
def mnistInfo():
    
    batch_size = 100

    #read mnist data
    mnist = input_data.read_data_sets(mnist_path,one_hot = True)

    #training data size
    print("Training data size ==> "+str(mnist.train.num_examples))

    #validating data size
    print("Validating data size ==> "+str(mnist.validation.num_examples))

    #testing data size
    print("Testing data size ==> "+str(mnist.test.num_examples))

    #traing data shape
    print("Shape of training images ==> "+str(mnist.train.images.shape))
    print("Shape of training labels ==> "+str(mnist.train.labels.shape))

    #print image
    print("Image ==> ")
    print(mnist.train.images[0])

    #print lable
    print("Lable ==> ")
    print(mnist.train.labels[0])

    #next batch size 
    xs,ys = mnist.train.next_batch(batch_size)
    print("X shape ==> "+str(xs.shape))
    print("Y shape ==> "+str(ys.shape))

def getmnist():
    mnist = input_data.read_data_sets(mnist_path,one_hot = True)
    return mnist

#Get current dir and execute file
def getcwd():
    print("Get current working dir ==> "+os.getcwd())
    print("Get current execute file ==> "+sys.argv[0])

def get_minst_class_num():
    #read mnist data
    mnist = input_data.read_data_sets(mnist_path,one_hot = True)
    
    labels = tf.placeholder(tf.float32,shape=[None,10],name="labels")
    class_tensor = tf.argmax(labels,axis=1)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print("class and num ==>")
        class_def,idx,count = sess.run(tf.unique_with_counts(class_tensor),feed_dict={labels:mnist.train.labels})
        print(class_def)
        print(count)


if __name__ == "__main__":
    #getcwd()
    mnistInfo()
    #get_minst_class_num()