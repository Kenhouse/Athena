










import tensorflow as tf

width = 224
height = 224
filter_1 = tf.Variable(tf.truncated_normal([11,11,3,96], mean=1.0, stddev=1.0), name='filter_1')
filter_2 = tf.Variable(tf.truncated_normal([5,5,96,256], mean=1.0, stddev=1.0), name='filter_2')
filter_3 = tf.Variable(tf.truncated_normal([3,3,256,384], mean=1.0, stddev=1.0), name='filter_3' )
filter_4 = tf.Variable(tf.truncated_normal([3,3,384,384], mean=1.0, stddev=1.0), name='filter_4' )
filter_5 = tf.Variable(tf.truncated_normal([3,3,384,256], mean=1.0, stddev=1.0), name='filter_5' )
bias_1 = tf.Variable(tf.zeros(shape=[96]),'bias_1')
bias_2 = tf.Variable(tf.zeros(shape=[256]),'bias_2')


def inference(input,
            number_of_class = 1000,
            is_training = True,
            dropout = 0.5):

    input = tf.reshape(input,[-1,width,height,3])
    conv1 = tf.nn.conv2d(input,filter_1,stride=[1,4,4,1],padding="SAME",name='conv1')
    conv1 = tf.nn.bias_add(conv1,bias_1)
