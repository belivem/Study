import tensorflow as tf

def get_softmax_logits(input_x,weigths1,weigths2,biases1,biases2,level):
    
    if level != 2:
        print("level is not right,it is limited 2 in function...")
        return
    
    #definelayer2
    layer1 = tf.nn.relu(tf.matmul(input_x,weigths1)+biases1)
    layer2 = tf.matmul(layer1,weigths2)+biases2
    return layer2

#Get the precision,recall,f1,mean_precision,mean_recall,mean_f1
def getMeasures(actual_labels,predict_labels,class_num):
    
    confusion_matrix = tf.confusion_matrix(actual_labels,predict_labels,num_classes=class_num)
    correct_predict = tf.diag_part(confusion_matrix)
    
    #precision
    colum_sum = tf.reduce_sum(confusion_matrix,axis=0)
    precision = tf.truediv(correct_predict,colum_sum)
    mean_precision = tf.reduce_mean(precision)
    
    #recall
    row_sum = tf.reduce_sum(confusion_matrix,axis=1)
    recall = tf.truediv(correct_predict,row_sum)
    mean_recall = tf.reduce_mean(recall)
    
    #f store
    f1 = tf.truediv((2*(tf.multiply(precision,recall))),tf.add(recall,precision))
    mean_f1 = tf.reduce_mean(f1)

    return (precision,recall,f1,mean_precision,mean_recall,mean_f1)

#load the persistence pb model
def load_pb_graph(filename):

    if not tf.gfile.Exists(filename):
        print(filename+" is not existd!")
        return None

    with tf.gfile.GFile(filename,mode="rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    tf.import_graph_def(graph_def,name="importPB")
    graph = tf.get_default_graph()
    return graph
