'''
MNIST number recognization Task.
1.Read MNIST data
2.Design Alexnet
3.Do training
4.Get training info
    - Validation accuracy for every 5 step.(Batch-normalization)
    - Get final accuracy
    - Add name and group for tensorboard
'''
import tensorflow as tf
import alexnet
from tensorflow.examples.tutorials.mnist import input_data



class MNISTRecognition:
    def __init__(self):
        return

    def inference(self, input):
        return



if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    alexnet.inference()
