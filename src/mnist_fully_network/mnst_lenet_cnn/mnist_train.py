import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference as infer
import numpy as np

MODEL_PATH = "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/"

BATCH_SIZE = 200
TRAINING_STEP = 20000
REGULARIZATION_RATE = 0.001
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.9
MOVING_AVERAGE_DECAY = 0.99

def train(mnist,num_examples):
    input_x = tf.placeholder(tf.float32,shape=[BATCH_SIZE,infer.INPUT_HIGHT,infer.INPUT_WIDTH,infer.INPUT_DEEPTH],name="input_x")
    input_y = tf.placeholder(tf.float32,shape=[BATCH_SIZE,infer.INPUT_LABELS],name="input_y")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    predict = infer.inference(input_x,True,regularizer)

    global_step = tf.Variable(0,trainable=False)

    #define the loss
    cross_entroy = tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=input_y)
    loss = tf.reduce_mean(cross_entroy) + tf.add_n(tf.get_collection("losses"))

    #define the moving average
    ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY,num_updates=global_step)
    moving_average_op = ema.apply(tf.trainable_variables())   

    #define the learning_rate and train_step
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step=global_step,decay_steps=num_examples/BATCH_SIZE,decay_rate=LEARNING_RATE_DECAY,name="learning_rate_decay")
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    train_op = tf.group(train_step,moving_average_op)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for  i in range(TRAINING_STEP):
            batch_x,batch_y = mnist.train.next_batch(BATCH_SIZE)
            reshape_batch_x = np.reshape(batch_x,(BATCH_SIZE,infer.INPUT_HIGHT,infer.INPUT_WIDTH,infer.INPUT_DEEPTH))
            sess.run(train_op,feed_dict={input_x:reshape_batch_x,input_y:batch_y})

            if i % 1000 == 0:
                batch_loss = sess.run(loss,feed_dict={input_x:reshape_batch_x,input_y:batch_y})
                print("After training "+str(i)+", the batch loss is "+str(batch_loss))

def getmnist(mnist_path):
    mnist = input_data.read_data_sets(mnist_path,one_hot = True)
    return mnist

if __name__ == "__main__":
    mnist_path = "/Users/liyanan/Documents/Test/Tensorflow/data/mnist_data/"
    mnist = getmnist(mnist_path)
    train_num_examples = mnist.train.num_examples
    train(mnist,train_num_examples)
