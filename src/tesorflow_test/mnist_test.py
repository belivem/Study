import tensorflow as tf
import mnist
import neural_util

mnist_path = "/Users/liyanan/Documents/Test/Tensorflow/src/tesorflow_test/data/mnist_data/"


def neural_netword():

    mnist_data = mnist.getmnist(mnist_path)
    #print the num_examples of training data
    print("Num of training data ==> "+str(mnist_data.train.num_examples))

    training_step = 20000   #训练轮数
    batch = 200
    input_size = 784
    hidden_size = 500
    output_size = 10
    
    #define input data
    input_x = tf.placeholder(tf.float32,shape=[None,input_size],name="input_x")
    input_y = tf.placeholder(tf.float32,shape=[None,output_size],name="input_y")

    #define parameters
    weigths1 = tf.Variable(tf.truncated_normal([input_size,hidden_size],dtype=tf.float32),name="weights1")
    weigths2 = tf.Variable(tf.truncated_normal([hidden_size,output_size],dtype=tf.float32),name="weights2")
    biases1 = tf.Variable(tf.fill([hidden_size],0.1),dtype=tf.float32,name="biases1")
    biases2 = tf.Variable(tf.fill([output_size],0.1),dtype=tf.float32,name="biases2")

    #get the logits layer  ==> softmax_y = tf.nn.softmax(layer2,name="softmax")
    layer2 = neural_util.get_softmax_logits(input_x,weigths1,weigths2,biases1,biases2,2)

    #define loss function  ==> cross_entroy = -tf.reduce_sum(input_y * tf.log(tf.clip_by_value(softmax_y,1e-10,1)),axis=1) 
    cross_entroy = tf.nn.softmax_cross_entropy_with_logits(logits=layer2,labels=input_y)
    loss = tf.reduce_mean(cross_entroy)

    #dfine training process
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    #define the measures
    layer2_output = neural_util.get_softmax_logits(input_x,weigths1,weigths2,biases1,biases2,2)
    correct_prediction = tf.equal(tf.argmax(layer2_output,1),tf.argmax(input_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    init = (tf.global_variables_initializer(),tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)

        for i in range(training_step):
            batch_x,batch_y = mnist_data.train.next_batch(batch)
            sess.run(train_op,feed_dict={input_x:batch_x,input_y:batch_y})

            if i % 1000 == 0:
                batch_loss_export = sess.run(loss,feed_dict={input_x:batch_x,input_y:batch_y})
                all_loss_exoprt = sess.run(loss,feed_dict={input_x:mnist_data.train.images,input_y:mnist_data.train.labels})
                batch_accuracy = sess.run(accuracy,feed_dict={input_x:batch_x,input_y:batch_y})
                all_accuracy = sess.run(accuracy,feed_dict={input_x:mnist_data.train.images,input_y:mnist_data.train.labels})
                print("After of "+str(i)+" training, the loss of batch_data is "+str(batch_loss_export)+", and the loass of all_data is "+str(all_loss_exoprt)+  
                ", the accuracy of batch_data is "+str(batch_accuracy)+", and the accuracy os all_data is "+str(all_accuracy))
                                


if __name__ == "__main__":
    neural_netword()