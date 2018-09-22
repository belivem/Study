import tensorflow as tf


Get the precision,recall,f1,mean_precision,mean_recall,mean_f1
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