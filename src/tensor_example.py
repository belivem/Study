from common.formats import image_format
from mnist_fully_network.mnist_practice import mnist_data_info

mnist_path = "/Users/liyanan/Documents/Test/Tensorflow/data/mnist_data/"

def test():
    mnist_data_info.mnistInfo()
    mnist = mnist_data_info.getmnist(mnist_path)

    images = mnist.train.images
    labels = mnist.train.labels
    print(type(images))
    print("Shape of training images ==> "+str(images.shape))
    print("Shape of training labels ==> "+str(labels.shape))

if __name__ == "__main__":
    test()
