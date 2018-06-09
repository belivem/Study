import tensorflow as tf
import mnist

mnist_path = "/Users/liyanan/Documents/Test/Tensorflow/src/tesorflow_test/data/mnist_data/"


def neural_netword():

    mnist_data = mnist.getmnist(mnist_path)
    #print the num_examples of training data
    print("Num of training data ==> "+str(mnist_data.train.num_examples))

    training_step = 30000   #训练轮数
    batch = 200
    input_node = 784
    hidden_node = 500
    output_node = 10
    
    #define input data
    input_x = tf.placeholder(tf.float32,shape=[None,input_node],name="input_x")
    input_y = tf.placeholder(tf.float32,shape=[None,output_node],name="input_y")

    #define parameters
    weigths1 = tf.Variable(tf.truncated_normal([input_node,hidden_node],tf.float32),name="weights1")
    weigths2 = tf.Variable(tf.truncated_normal([hidden_node,output_node],tf.float32),name="weights2")
    biases1 = tf.Variable(tf.fill([hidden_node],0.1),name="biases1")
    biases2 = tf.Variable(tf.fill([output_node],0.1),name="biases2")

    #define neural network structure
    layer1 = tf.nn.relu(tf.matmul(input_x,weigths1)+biases1,name="layer1")
    layer2 = tf.nn.relu(tf.matmul(layer1,weigths2)+biases2,name="layer2")
    softmax_y = tf.nn.softmax(layer2,name="softmax")

    #define loss function
    cross_entroy = -tf.reduce_sum(input_y * tf.log(tf.clip_by_value(softmax_y,1e-10,1)))
    loss = tf.reduce_mean(cross_entroy)

    #dfine training process
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    init = (tf.global_variables_initializer(),tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)

        for i in range(training_step):
            batch_x,batch_y = mnist_data.train.next_batch(batch)
            sess.run(train_op,feed_dict={input_x:batch_x,input_y:batch_y})

            if i % 1000 == 0:
                loss_exoprt = sess.run(loss,feed_dict={input_x:mnist_data.train.images,input_y:mnist_data.train.labels})
                print("After of "+i+" training, the loss of all_data is "+loss_exoprt)

if __name__ == "__main__":
    neural_netword()