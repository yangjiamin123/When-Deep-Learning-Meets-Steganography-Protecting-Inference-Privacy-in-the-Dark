#GHOST+：MNIST dataset
# This code can be used in any of the following cases: 1-9,2-8,3-7,4-6, but the data processing is different

import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def read_csv(filename):
    file_name=pd.read_csv(filename,header=None,index_col=None,skiprows=0,sep=',')
    image=file_name.iloc[:,1:785]
    label=file_name.iloc[:,0]
    label_batch = tf.one_hot(label, depth=10)

    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [-1, 10])

    image=tf.cast(image, tf.float32) * (1. / 255)
    return image,label_batch

def get_batch(data,label,size):
    start_index=np.random.randint(0,data.shape[1]-size)
    return data[start_index:(start_index+size),:],label[start_index:(start_index+size),:]

train_image, train_label=read_csv('lsbmnist3-7train.csv')
test_image,test_labels=read_csv('mnist1-9test.csv')
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)
def fc_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
def addConnect(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.01))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # s = tf.layers.BatchNormalization(Wx_plus_b)
    if activation_function is None:
        return Wx_plus_b
    else:
        return activation_function(Wx_plus_b)
def Generator(X):
    with tf.variable_scope('G'):
        connect_1 = addConnect(X, 784, 256, tf.nn.relu)
        connect_2 = addConnect(connect_1, 256, 512, tf.nn.relu)
        connect_3 = addConnect(connect_2, 512, 1024, tf.nn.relu)
        g_output=addConnect(connect_3, 1024, 784, tf.nn.tanh)

        g_output=tf.reshape(g_output, (-1,28,28,1), name=None)
        print(g_output.shape)
        print("!!!!!!!!!!!!!!")
    return g_output

def Discriminator(data):
    with tf.variable_scope('C'):
        convFilter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], mean=0, stddev=0.1))
        convBias1 = tf.Variable(tf.truncated_normal([32], mean=0, stddev=0.1))
        convLayer1 = tf.nn.conv2d(input=data, filter=convFilter1, strides=[1, 1, 1, 1], padding='SAME')
        convLayer1 = tf.add(convLayer1, convBias1)
        convLayer1 = tf.nn.relu(convLayer1)
        poolLayer1 = tf.nn.max_pool(value=convLayer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        convFilter2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], mean=0, stddev=0.1))
        convBias2 = tf.Variable(tf.truncated_normal([64], mean=0, stddev=0.1))
        convLayer2 = tf.nn.conv2d(input=poolLayer1, filter=convFilter2, strides=[1, 1, 1, 1], padding='SAME')
        convLayer2 = tf.add(convLayer2, convBias2)
        convLayer2 = tf.nn.relu(convLayer2)
        poolLayer2 = tf.nn.max_pool(value=convLayer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        fullWeight = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 64, 1024], mean=0, stddev=0.1))
        fullBias = tf.Variable(tf.truncated_normal(shape=[1024], mean=0.0, stddev=0.1))
        fullInput = tf.reshape(poolLayer2, [-1, 7 * 7 * 64])
        fullLayer = tf.add(tf.matmul(fullInput, fullWeight), fullBias)
        fullLayer = tf.nn.relu(fullLayer)
        outputWeight = tf.Variable(tf.truncated_normal(shape=[1024, 10], mean=0.0, stddev=0.1))
        outputBias = tf.Variable(tf.truncated_normal(shape=[10], mean=0, stddev=0.1))
        result = tf.add(tf.matmul(fullLayer, outputWeight), outputBias)

        prob = tf.nn.softmax(result)
        return result, prob

x= tf.placeholder(tf.float32,shape=[None, 784])
y = tf.placeholder(tf.float32, [None, 10])
gen= Generator(x)
print(gen.shape)
result, prob = Discriminator(gen)

loss_g = tf.reduce_mean(tf.square(x-gen))
loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=result))
loss=0.1*loss_g+0.9*loss_c
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prob, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
params_g = tf.global_variables(scope='G')
params_c = tf.global_variables(scope='C')
saver_g = tf.train.Saver(var_list=params_g)
saver_c = tf.train.Saver(var_list=params_c)
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver=tf.train.Saver(var_list=params_c)
init = tf.global_variables_initializer()

num_samples=60000
batch_size=256
n_batch=num_samples//batch_size
model='train'
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    if model=="train":
        for epoch in range(20):
            for i in range(n_batch):
                batch_xs, batch_ys = get_batch(train_image,train_label,batch_size)

                example,l=sess.run([batch_xs,batch_ys])
                print(example.shape)
                print(example)
                example=example.reshape(-1,28,28,1)
                print(example.shape)
                _, train_loss, acc = sess.run([train_step, loss, accuracy], feed_dict={x: example, y: l})
                print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
                print(train_loss)
            # saver.save(sess, "Model_g/model.ckpt")
        example, l = sess.run([test_image, test_labels])
        acc = sess.run([accuracy], feed_dict={x: example, y: l})
        # print(np.array(gen).shape)
        print('test accuracy:',acc)