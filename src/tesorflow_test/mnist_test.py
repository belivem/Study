import tensorflow as tf
import mnist
import neural_util

mnist_path = "/Users/liyanan/Documents/Test/Tensorflow/src/tesorflow_test/data/mnist_data/"
mnist_fully_connected_persistence_path = "/Users/liyanan/Documents/Test/Tensorflow/models/model_pb/model_minst/mnist_fully_connected_model.pb"

def fully_connected_networt():

    mnist_data = mnist.getmnist(mnist_path)
    #print the num_examples of training data
    print("Num of training data ==> "+str(mnist_data.train.num_examples))

    training_step = 25000           #训练轮数
    batch = 200                     #batch值
    input_size = 784                #输入层节点个数
    hidden_size = 500               #隐藏层节点个数
    output_size = 10                #输出层节点个数
    regularization_rate = 0.001    #l2正则化参数
    learning_rate_base = 0.8        #学习率初始值
    learning_rate_decay = 0.99      #学习率衰减率
    moving_average_decay = 0.99     #滑动平均模型衰减率

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
    l2_regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)(weigths1) + tf.contrib.layers.l2_regularizer(regularization_rate)(weigths2)
    loss = tf.reduce_mean(cross_entroy)+l2_regularizer

    #dfine training process ,set decay learning_rate[指数衰减法] and moving_average[滑动平均模型]
    #set decay learning rate
    global_steps = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate_base,global_step=global_steps,decay_steps=100,decay_rate=learning_rate_decay,name="learning_rate_dynamic")
    
    #set moving average
    ema = tf.train.ExponentialMovingAverage(decay=moving_average_decay,num_updates=global_steps)
    moving_average_op = ema.apply(tf.trainable_variables())

    #set train operation
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_steps)
    train_op = tf.group(train_step,moving_average_op)

    #Get the measures
    #layer2_output = neural_util.get_softmax_logits(input_x,weigths1,weigths2,biases1,biases2,2)
    #correct_prediction = tf.equal(tf.argmax(layer2_output,1),tf.argmax(input_y,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    actual_labels = tf.argmax(input_y,1)
    predict_labels = tf.argmax(neural_util.get_softmax_logits(input_x,weigths1,weigths2,biases1,biases2,2),1)
    (precision,recall,f1,mean_precision,mean_recall,mean_f1) = neural_util.getMeasures(actual_labels,predict_labels)

    init = (tf.global_variables_initializer(),tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)

        for i in range(training_step):
            batch_x,batch_y = mnist_data.train.next_batch(batch)
            sess.run(train_op,feed_dict={input_x:batch_x,input_y:batch_y})

            if i % 1000 == 0:
                batch_loss_export = sess.run(loss,feed_dict={input_x:batch_x,input_y:batch_y})
                all_loss_exoprt = sess.run(loss,feed_dict={input_x:mnist_data.train.images,input_y:mnist_data.train.labels})

                measures = sess.run((precision,recall,f1,mean_precision,mean_recall,mean_f1),feed_dict={input_x:mnist_data.validation.images,input_y:mnist_data.validation.labels})

                print("After of "+str(i)+" training, the loss of batch_data is "+str(batch_loss_export)+", and the loss of all_data is "+str(all_loss_exoprt)+  
                ", mean_precision is "+str(measures[3])+", mean_recall is "+str(measures[4])+", mean_f1 is "+str(measures[5]))

            if i == (training_step - 1):
                print("After of "+str(i)+" training, the loss of batch_data is "+str(batch_loss_export)+", and the loass of all_data is "+str(all_loss_exoprt)+  
                    ", precision is "+str(measures[0])+", recall is "+str(measures[1])+", f1 is "+str(measures[2])+", mean_precision is "+str(measures[3])+", mean_recall is "
                    +str(measures[4])+", mean_f1 is "+str(measures[5]))                    
        
        #Persist fully connected network for mnist
        print("Persist fully connected ==> ")
        graph_def = tf.get_default_graph().as_graph_def()  #Get the current graph's GraphDef
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess,graph_def,output_node_names=['add_1'])
        with tf.gfile.GFile(mnist_fully_connected_persistence_path,mode="wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("persistence is done!")


#load the fully connected network
def load_mnist_fully_connected_network():

    output_size = 10;
    mnist_data = mnist.getmnist(mnist_path)

    graph = neural_util.load_pb_graph(mnist_fully_connected_persistence_path)
    if graph == None:
        print("load PB model has error!")
        return

    #for op in graph.get_operations():
    #    print(op.name,op.values())

    #input tensor and output tensor
    input_x = graph.get_tensor_by_name("importPB/input_x:0")
    input_y = tf.placeholder(tf.float32,shape=[None,output_size],name="input_y")

    output_y = graph.get_tensor_by_name("importPB/add_1:0")
    
    actual_labels = tf.argmax(input_y,1)
    predict_labels = tf.argmax(output_y,1)
    (precision,recall,f1,mean_precision,mean_recall,mean_f1) = neural_util.getMeasures(actual_labels,predict_labels,output_size)
    
    with tf.Session() as sess:
        measures = sess.run((precision,recall,f1,mean_precision,mean_recall,mean_f1),feed_dict={input_x:mnist_data.test.images,input_y:mnist_data.test.labels})        
        print("precision is "+str(measures[0])+", recall is "+str(measures[1])+", f1 is "+str(measures[2])+", mean_precision is "+str(measures[3])+", mean_recall is "
                    +str(measures[4])+", mean_f1 is "+str(measures[5]))  

if __name__ == "__main__":
    load_mnist_fully_connected_network()
