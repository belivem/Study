import tensorflow as tf


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