import tensorflow as tf
import mnist_inference
import mnist_train
import neural_util

model_list = ["/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-15001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-16001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-17001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-18001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-19001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-20001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-21001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-22001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-23001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-24001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-25001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-26001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-27001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-28001",
              "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-29001"]

class Perdic:

    def __init__(self,ckpt_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_x = tf.placeholder(tf.float32,shape=[None,mnist_inference.INPUT_SIZE],name="input_x")
            self.input_y = tf.placeholder(tf.float32,shape=[None,mnist_inference.OUTPUT_SIZE],name="input_y")

            predict_y = mnist_inference.construct_network(self.input_x,None)

            actual_labels  = tf.argmax(self.input_y,1) 
            predict_labels = tf.argmax(predict_y,1)

            self.measures = neural_util.getMeasures(actual_labels,predict_labels,mnist_inference.OUTPUT_SIZE)
            ema = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
            self.Saver = tf.train.Saver(ema.variables_to_restore())

        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.Saver.restore(self.sess,ckpt_path)
        
    def test(self,steps,mnist):
        measures = self.sess.run(self.measures,feed_dict={self.input_x:mnist.test.images,self.input_y:mnist.test.labels})
        print("After of "+steps+" training, the mean_precision is "+str(measures[3])+", mean_recall is "+str(measures[4])+", mean_f1 is "+str(measures[5]))
        self.sess.close()

if __name__ == "__main__":
    mnist_path = "/Users/liyanan/Documents/Test/Tensorflow/data/mnist_data/"
    mnist_data = mnist_train.getmnist(mnist_path)

    for model_path in model_list:
        steps = model_path.split("-")[-1]
        instance = Perdic(model_path)
        instance.test(steps,mnist_data)

    #instance = Perdic("/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/mnist_model.ckpt-18001")
    #instance.test("18001",mnist_data)