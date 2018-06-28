import tensorflow as tf

INPUT_SIZE = 784
HIDDEN_SIZE = 500
OUTPUT_SIZE = 10

#Get the weights variable
def get_weight_variables(shape,regularizer):
    weights = tf.get_variable("weights",shape=shape,dtype=tf.float32,initializer=tf.truncated_normal_initializer())

    if regularizer != None:
        tf.add_to_collection("loss",regularizer(weights))

    return weights

#Construct the neutual network
def construct_network(input_tensor,regularizer):
    #the layer1
    with tf.variable_scope("layer1"):
        weights = get_weight_variables([INPUT_SIZE,HIDDEN_SIZE],regularizer)
        biases = tf.Variable(tf.fill([HIDDEN_SIZE],0.1),dtype=tf.float32,name="biases")
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
    #the layer2
    with tf.variable_scope("layer2"):
        weighsts = get_weight_variables([HIDDEN_SIZE,OUTPUT_SIZE],regularizer)
        biases = tf.Variable(tf.fill([OUTPUT_SIZE],0.1),dtype=tf.float32,name="biases")
        layer2 = tf.matmul(layer1,weighsts) + biases

    return layer2    
