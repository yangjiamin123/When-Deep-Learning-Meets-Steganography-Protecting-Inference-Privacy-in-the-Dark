#GHOST：Retraining the network with MNIST dataset
# This code can be used in any of the following cases: 1-9,2-8,3-7,4-6, but the data processing is different
import time

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import numpy as np
from GHOST.net_local import local_mnist



def read_csv(filename):
    file_name=pd.read_csv(filename,header=None,index_col=None,skiprows=0,sep=',')
    image=file_name.iloc[:,1:785]
    label=file_name.iloc[:,0]
    label_batch = tf.one_hot(label, depth=10)

    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [-1, 10])

    image=tf.cast(image, tf.float32) * (1.0 / 255)
    return image,label_batch
def get_batch(data,label,size):
    start_index=np.random.randint(0,data.shape[1]-size)
    return data[start_index:(start_index+size),:],label[start_index:(start_index+size),:]

train_image, train_label=read_csv('../../dataset/MNIST/CSV/non-lsbmnist2-8train.csv')
test_image,test_label=read_csv('../../dataset/MNIST/CSV/mnist1-9test.csv')
#test_image,test_label=read_csv('../../dataset/MNIST/CSV/mnist_test.csv')
print(train_image.shape)
print('...........')
x= tf.placeholder(tf.float32,shape=[None, 28,28,1])
y= tf.placeholder(tf.float32, [None, 10])

local_param=local_mnist(x)

convFilter2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], mean=0, stddev=0.1))
convBias2 = tf.Variable(tf.truncated_normal([64], mean=0, stddev=0.1))
convLayer2 = tf.nn.conv2d(input=local_param, filter=convFilter2, strides=[1, 1, 1, 1], padding='SAME')
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

loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=result))
target = tf.train.AdamOptimizer().minimize(loss)

startTime = time.time()
def get_batch(data,label,size):
    start_index=np.random.randint(0,data.shape[1]-size)
    return data[start_index:(start_index+size),:],label[start_index:(start_index+size),:]
# train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batchSize = 256
    for i in range(20):
        batch_xs, batch_ys = get_batch(train_image, train_label, batchSize)
        example, l = sess.run([batch_xs, batch_ys])
        example = example.reshape(-1, 28, 28, 1)
        sess.run([target, loss], feed_dict={x: example, y: l})

        corrected = tf.equal(tf.argmax(y, 1), tf.argmax(result, 1))
        accuracy = tf.reduce_mean(tf.cast(corrected, tf.float32))
        accuracyValue = sess.run(accuracy, feed_dict={x: example, y: l})
        print(i, 'train set accuracy:', accuracyValue)

    endTime = time.time()
    print('train time:', endTime - startTime)

    corrected  = tf.equal(tf.argmax(y, 1), tf.argmax(result, 1))
    accuracy   = tf.reduce_mean(tf.cast(corrected, tf.float32))
    example, l = sess.run([test_image, test_label])
    accuracyValue = sess.run(accuracy, feed_dict={x: example, y: l})
    print("accuracy on test set:", accuracyValue)
    sess.close()
