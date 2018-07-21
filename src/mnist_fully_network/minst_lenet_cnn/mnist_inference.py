import tensorflow as tf

#input_data
INPUT_HIGHT = 28
INPUT_WIDTH = 28
INPUT_DEEPTH = 1
INPUT_LABELS = 10
#cov1
COV1_SIZE = 5
COV1_DEEPTH = 32

POOL1_SIZE = 2
#cov2
COV2_SIZE = 5
COV2_DEEPTH = 64

POOL2_SIZE = 2
#fully_connect1
FULLY_C1 = 512


def inference(input,train,regularizer):

    #Convolutional Layer1  input: 28*28*1  conv: 5*5*32  output: 14*14*32 
    with tf.variable_scope("cov1"):
        conv1_weights = tf.get_variable("conv1_weights",shape=[COV1_SIZE,COV1_SIZE,INPUT_DEEPTH,COV1_DEEPTH],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("conv1_biases",shape=[COV1_DEEPTH],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input,conv1_weights,strides=[1,1,1,1],padding='SAME')
        conv1_result = tf.nn.relu(conv1 + conv1_biases)

        conv1_pool = tf.nn.max_pool(conv1_result,ksize=[1,POOL1_SIZE,POOL1_SIZE,1],strides=[1,2,2,1],padding='SAME')

    #Convolutional Layer2 input: 14*14*32 conv: 5*5*64  output: 7*7*64
    with tf.variable_scope("conv2"):
        conv2_weights = tf.get_variable("conv2_weights",shape=[COV2_SIZE,COV2_SIZE,COV1_DEEPTH,COV2_DEEPTH],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("conv2_biases",shape=[COV2_DEEPTH],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(conv1_pool,conv2_weights,strides=[1,1,1,1],padding="SAME")
        conv2_result = tf.nn.relu(conv2 + conv2_biases)

        conv2_pool = tf.nn.max_pool(conv2_result,ksize=[1,POOL2_SIZE,POOL2_SIZE,1],strides=[1,2,2,1],padding="SAME")

    #TransForm conv matrixs to nodes list
        #batch_size = conv2_pool.get_shape()[0].value
        input_hight = conv2_pool.get_shape()[1].value
        input_width = conv2_pool.get_shape()[2].value
        input_deepth = conv2_pool.get_shape()[3].value
        nodes = input_hight*input_width*input_deepth
        fc_input = tf.reshape(conv2_pool,[-1,nodes])

        #pool_shape = conv2_pool.get_shape().as_list()
        #nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        #reshaped = tf.reshape(conv2_pool, [pool_shape[0], nodes])
        
    #FULLY_CONNECT LAYER1
    with tf.variable_scope("fully_connetc1"):
        fc1_weights = tf.get_variable("fc1_weights",shape=[nodes,FULLY_C1],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases = tf.get_variable("fc1_biases",shape=[FULLY_C1],initializer=tf.constant_initializer(0.0))
        fc1_result = tf.nn.relu(tf.matmul(fc_input,fc1_weights) + fc1_biases)

        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fc1_weights))

        #if train:
        #    fc1_result = tf.nn.dropout(fc1_result,0.5)

    #FULLY_CONNECT LAYER2
    with tf.variable_scope("fully_connetc2"):
        fc2_weights = tf.get_variable("fc2_weights",shape=[FULLY_C1,INPUT_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable("fc2_biases",shape=[INPUT_LABELS],initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(fc1_result,fc2_weights) + fc2_biases
    
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fc2_weights))

    return logits