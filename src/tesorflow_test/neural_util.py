import tensorflow as tf

def get_softmax_logits(input_x,weigths1,weigths2,biases1,biases2,level):
    
    if level != 2:
        print("level is not right,it is limited 2 in function...")
        return
    
    #definelayer2
    layer1 = tf.nn.relu(tf.matmul(input_x,weigths1)+biases1)
    layer2 = tf.matmul(layer1,weigths2)+biases2
    return layer2