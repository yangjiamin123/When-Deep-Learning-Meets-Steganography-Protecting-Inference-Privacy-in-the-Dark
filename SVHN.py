#GHOST+：SVHN dataset
# This code can be used in any of the following cases: 1-9,2-8,3-7,4-6, but the data processing is different
import tensorflow as tf
#Die Hauptbibliothek Tensorflow wird geladen
import tensorflow as tf
from sklearn.utils import shuffle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import csv
import pandas as pd
from tensorflow.contrib.layers import flatten
Trainingsbilder = []
Trainingslabels = []
print("loading training dataset")
name=[3,4,5]
for i in name:
    n = str(i)
    Pfad = "D:\\dataset\\G3-7\\train\\" + n
    label=i
    for Datei in os.listdir(Pfad):
        img = os.path.join(Pfad,Datei)
        #print(img)
        img = image.load_img(img,target_size=(32,32))
        img = image.img_to_array(img,  dtype=np.float32)
        img=img.reshape(1,32,32,3)
        Trainingsbilder.append(img)
        Trainingslabels.append(label)


#Convert list to tensor
Trainingslabels = np.asarray(Trainingslabels)
Trainingsbilder = np.asarray([Trainingsbilder])
Trainingsbilder = Trainingsbilder.reshape(-1, 32, 32, 3)
#0-1
Trainingsbilder = Trainingsbilder/255
X_train = np.asarray(Trainingsbilder, dtype = "float32")
Trainingslabels = np.asarray(Trainingslabels, dtype= "float32")
print(X_train.shape)
print(Trainingslabels)
X_train, y_train = shuffle(X_train, Trainingslabels)

y_onehot=tf.one_hot(y_train,10)
## Take the training dataset at random
sel = np.random.choice(X_train.shape[0], size=100, replace=False)
X_train_test = X_train[sel,:]
y_train_test = y_train[sel]
y_train_test = tf.one_hot(y_train_test,43)

Testbilder = []
Testlabels = []
print("loading test dataset")

#Laden der Testbilder  als deren Bildtensoren in eine Liste
Testpfad="D:\\dataset\\G3-7\\test\\"
for Datei in os.listdir(Testpfad):
     #print(Datei)
     img = os.path.join(Testpfad,Datei)
     img = image.load_img(img,target_size=(32,32))
     img = image.img_to_array(img,  dtype=np.float32)
     img = img.reshape(1,32,32, 3)
     Testbilder.append(img)


with open('test.csv') as csvdatei:
    csv_datei = csv.reader(csvdatei)
    for Reihe in csv_datei:
        Testlabels.append(Reihe[6])

Testlabels.pop(0)
Testlabels = np.asarray(Testlabels)
Testbilder = np.asarray([Testbilder])
Testbilder = Testbilder.reshape(-1, 32, 32, 3)
#Umwandlung der Farbwerte in Gleitkommazahlen zwischen 0 und 1
Testbilder = Testbilder/255
X_test = np.asarray(Testbilder, dtype = "float32")
Testlabels = np.asarray(Testlabels, dtype= "float32")
print(Testlabels.shape)
print(X_test.shape)
print('////')
X_test, y_test = shuffle(X_test, Testlabels)
y_onehot2=tf.one_hot(y_test,10)
print(X_test.shape)
print(y_onehot2.shape)
print(Testlabels)
def ConvInstNormRelu(x, filters, kernel_size=3, strides=1):
	Conv = tf.layers.conv2d(
						inputs=x,
						filters=filters,
						kernel_size=kernel_size,
						strides=strides,
						padding="same",
						activation=None)

	InstNorm = tf.contrib.layers.instance_norm(Conv)

	return tf.nn.relu(InstNorm)
# helper function for trans convolution -> instance norm -> relu
def TransConvInstNormRelu(x, filters, kernel_size=3, strides=2):
	TransConv = tf.layers.conv2d_transpose(
						inputs=x,
						filters=filters,
						kernel_size=kernel_size,
						strides=strides,
						padding="same",
						activation=None)

	InstNorm = tf.contrib.layers.instance_norm(TransConv)

	return tf.nn.relu(InstNorm)


def ResBlock(x, training, filters=32, kernel_size=3, strides=1):
	conv1 = tf.layers.conv2d(
						inputs=x,
						filters=filters,
						kernel_size=kernel_size,
						strides=strides,
						padding="same",
						activation=None)

	conv1_norm = tf.layers.batch_normalization(conv1, training=training)

	conv1_relu = tf.nn.relu(conv1_norm)

	conv2 = tf.layers.conv2d(
						inputs=conv1_relu,
						filters=filters,
						kernel_size=kernel_size,
						strides=strides,
						padding="same",
						activation=None)

	conv2_norm = tf.layers.batch_normalization(conv2, training=training)


	return x + conv2_norm

def Generator(X,training):
    with tf.variable_scope('G'):
        c1 = ConvInstNormRelu(X, filters=8, kernel_size=3, strides=1)
        print('c1:',c1.shape)
        d1 = ConvInstNormRelu(c1, filters=16, kernel_size=3, strides=2)
        print('d1:',d1.shape)
        d2 = ConvInstNormRelu(d1, filters=32, kernel_size=3, strides=2)
        print('d2:',d2.shape)
        rb1 = ResBlock(d2, training, filters=32)
        print('rb1:',rb1.shape)
        rb2 = ResBlock(rb1, training, filters=32)
        print('rb2:',rb2.shape)
        rb3 = ResBlock(rb2, training, filters=32)
        print('rb3:',rb3.shape)
        rb4 = ResBlock(rb3, training, filters=32)
        print('rb4:',rb4.shape)

        # upsample using conv transpose
        u1 = TransConvInstNormRelu(rb4, filters=16, kernel_size=3, strides=2)
        print('u1:',u1.shape)
        u2 = TransConvInstNormRelu(u1, filters=8, kernel_size=3, strides=2)
        print('u2:',u2.shape)

        # final layer block
        out = tf.layers.conv2d_transpose(
            inputs=u2,
            filters=x.get_shape()[-1].value,  # or 3 if RGB image
            kernel_size=3,
            strides=1,
            padding="same",
            activation=None)

        # out = tf.contrib.layers.instance_norm(out)
        print(out.shape)

        return tf.nn.tanh(out)

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

def avg_pool_2by2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def global_avg_pool(x):
    return tf.nn.avg_pool(x, ksize=[1, 8, 8, 1],
                          strides=[1, 8, 8, 1], padding='SAME')

def norm(x):
    return tf.nn.lrn(x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def conv_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def fc_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
def AlexNet(x, KEEP_PROB, LAMBDA):
    with tf.variable_scope('D'):
        mu = 0
        sigma = 0.1
        reg_loss = tf.zeros(1)
        conv1_w = tf.Variable(tf.truncated_normal((5, 5, 3, 9), mu, sigma))
        conv1_b = tf.Variable(tf.zeros(9))
        conv1 = tf.nn.conv2d(x, conv1_w, [1, 1, 1, 1], 'VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        conv2_w = tf.Variable(tf.truncated_normal((3, 3, 9, 32), mu, sigma))
        conv2_b = tf.Variable(tf.zeros(32))
        conv2 = tf.nn.conv2d(pool1, conv2_w, [1, 1, 1, 1], 'VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        conv3_w = tf.Variable(tf.truncated_normal((3, 3, 32, 48), mu, sigma))
        conv3_b = tf.Variable(tf.zeros(48))
        conv3 = tf.nn.conv2d(pool2, conv3_w, [1, 1, 1, 1], 'SAME') + conv3_b
        conv3 = tf.nn.relu(conv3)
        conv4_w = tf.Variable(tf.truncated_normal((3, 3, 48, 64), mu, sigma))
        conv4_b = tf.Variable(tf.zeros(64))
        conv4 = tf.nn.conv2d(conv3, conv4_w, [1, 1, 1, 1], 'SAME') + conv4_b
        conv4 = tf.nn.relu(conv4)
        conv5_w = tf.Variable(tf.truncated_normal((3, 3, 64, 96), mu, sigma))
        conv5_b = tf.Variable(tf.zeros(96))
        conv5 = tf.nn.conv2d(conv4, conv5_w, [1, 1, 1, 1], 'SAME') + conv5_b
        conv5 = tf.nn.relu(conv5)
        pool3 = tf.nn.max_pool(conv5, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        flat = flatten(pool3)
        full1_w = tf.Variable(tf.truncated_normal((864, 400), mu, sigma))
        full1_b = tf.Variable(tf.zeros(400))
        full1 = tf.matmul(flat, full1_w) + full1_b
        full1 = tf.nn.relu(full1)
        full1 = tf.nn.dropout(full1, KEEP_PROB)
        full2_w = tf.Variable(tf.truncated_normal((400, 160), mu, sigma))
        full2_b = tf.Variable(tf.zeros(160))
        full2 = tf.matmul(full1, full2_w) + full2_b
        full2 = tf.nn.relu(full2)
        full2 = tf.nn.dropout(full2, KEEP_PROB)
        full3_w = tf.Variable(tf.truncated_normal((160, 10), mu, sigma))
        full3_b = tf.Variable(tf.zeros(10))
        logits = tf.matmul(full2, full3_w) + full3_b
        if LAMBDA != 0:
            reg_loss = tf.nn.l2_loss(conv1_w) + tf.nn.l2_loss(conv2_w) + tf.nn.l2_loss(conv3_w) + tf.nn.l2_loss(
                conv4_w) + tf.nn.l2_loss(conv5_w) + tf.nn.l2_loss(full1_w) + tf.nn.l2_loss(full2_w) + tf.nn.l2_loss(full3_w)

        return logits, reg_loss



model = "train"
x= tf.placeholder(tf.float32, [None, 32,32,3])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder_with_default(1.0, shape=())
LAMBDA = tf.placeholder_with_default(0.0, shape=())
is_training=tf.placeholder(tf.bool,[])

gen= Generator(x,is_training)
print('x:',x.shape)
print('gen:',gen.shape)

logits,reg_loss = AlexNet(gen,keep_prob, LAMBDA)
print(logits.shape)

loss_g = tf.reduce_mean(tf.square(x-gen))
loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
loss=loss_c+loss_g
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
y_pred = tf.argmax(logits, 1)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
params_g = tf.global_variables(scope='G')
params_c = tf.global_variables(scope='D')
saver_g = tf.train.Saver(var_list=params_g)
saver_c = tf.train.Saver(var_list=params_c)
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

num_samples=25200
batch_size=128
n_batch=num_samples//batch_size
test_accuracy=[]

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver_c.restore(sess, "./model/alexnet.ckpt")
    if model=='train':
        for epoch in range(50):
            for i in range(n_batch):
                example=X_train[i*batch_size:(i+1)*batch_size]
                l=y_onehot[i*batch_size:(i+1)*batch_size]
                label=sess.run(l)
                _train_step, train_loss, acc= sess.run([train_step, loss, accuracy], feed_dict={x: example, y: label,is_training:True})
                print("Iter " + str(epoch) + ",Training Accuracy:" + str(acc))
                print(train_loss)
        label = sess.run(y_onehot2)
        img,acc=sess.run([gen,accuracy],feed_dict={x:X_test,y:label,is_training:False})
        print('test accuracy:',acc)
        saver_g.save(sess, "SVHNmodel_g/model.ckpt")

    else:
        saver_g.restore(sess, "./SVHNmodel_g/model.ckpt")
        saver_c.restore(sess, "./model/alexnet.ckpt")

        y_train_test_label = sess.run(y_train_test)
        train_loss2, acc2, _y_pred2, _loss_g2 = sess.run([loss, accuracy, y_pred, loss_g],
                                                         feed_dict={x: X_train_test, y: y_train_test_label,
                                                                    is_training: True})
        with open('train_sample_valid.txt', 'a+') as f:
            f.write(str(_y_pred2) + '\n')
        print(str(acc2))
        print(_loss_g2)
        print(train_loss2)





