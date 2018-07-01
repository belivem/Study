
import tensorflow as tf
MODEL_PATH = "/Users/liyanan/Documents/Test/Tensorflow/models/model_ckpt/model_minst/"


def test():
  a = tf.Variable(1.0,dtype=tf.float32,name="a")
  b = tf.Variable(2.0,dtype=tf.float32,name="b")
  result = tf.add(a,b,name="add")

  
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
      sess.run(init)
      ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
      print(sess.run(ckpt))
      print(sess.run(result))

if __name__ == "__main__":
    test()