#GHOST：Retraining the network with MNIST dataset
# This code can be used in any of the following cases: 1-9,2-8,3-7,4-6, but the data processing is different
import tensorflow as tf
#Die Hauptbibliothek Tensorflow wird geladen
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adam
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import csv
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

Trainingsbilder = []
Trainingslabels = []
print("loading training dataset")
name=[1,2,3,4,5,6,7,8,9,10]
for i in name:
    n = str(i)
    Pfad = "D:\\dataset\\svhn-original\\" + n
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
X_train, y_train = shuffle(X_train, Trainingslabels)
y_onehot=tf.one_hot(y_train,10)
print(X_train.shape)
print(y_onehot.shape)

Testbilder = []
Testlabels = []
print("loading test dataset")
name=[1,2,3,4,5,6,7,8,9,10]
for i in name:
    n = str(i)
    Pfad = "D:\\dataset\\svhn-original-test\\" + n
    label=i
    for Datei in os.listdir(Pfad):
        img = os.path.join(Pfad,Datei)
        #print(img)
        img = image.load_img(img,target_size=(32,32))
        img = image.img_to_array(img,  dtype=np.float32)
        img=img.reshape(1,32,32,3)
        Testbilder.append(img)
        Testlabels.append(label)


#Convert list to tensor
Testlabels = np.asarray(Testlabels)
Testbilder = np.asarray([Testbilder])
Testbilder = Testbilder.reshape(-1, 32, 32, 3)
#0-1
Testbilder = Testbilder/255
X_test = np.asarray(Testbilder, dtype = "float32")
Testlabels = np.asarray(Testlabels, dtype= "float32")
X_test, y_test = shuffle(X_test, Testlabels)
y_onehot2=tf.one_hot(y_test,10)
print(X_test.shape)
print(y_onehot2.shape)


def AlexNet(x, KEEP_PROB, LAMBDA):
    with tf.variable_scope('SVHN'):
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
        full3_w = tf.Variable(tf.truncated_normal((160, 11), mu, sigma))
        full3_b = tf.Variable(tf.zeros(11))
        logits = tf.matmul(full2, full3_w) + full3_b

        if LAMBDA != 0:
            reg_loss = tf.nn.l2_loss(conv1_w) + tf.nn.l2_loss(conv2_w) + tf.nn.l2_loss(conv3_w) + tf.nn.l2_loss(
                conv4_w) + tf.nn.l2_loss(conv5_w) + tf.nn.l2_loss(full1_w) + tf.nn.l2_loss(full2_w) + tf.nn.l2_loss(full3_w)

        return logits, reg_loss

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None,10))
keep_prob = tf.placeholder_with_default(1.0, shape=())
LAMBDA = tf.placeholder_with_default(0.0, shape=())

# Hyperparameters
LEARNING_RATE = 5e-4
EPOCHS =50
BATCH_SIZE = 128

logits, reg_loss = AlexNet(x, keep_prob, LAMBDA)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss_op = tf.reduce_mean(cross_entropy)
# batch gradient descent optimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
# Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
train_op = optimizer.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_op, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


train_accuracy = []
test_accuracy = []
learning_rates = []
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    for i in range(EPOCHS):
        total_train_acc = 0
        print("EPOCH {} :".format(i + 1), end=' ')
        for offset in range(0, num_examples, BATCH_SIZE):  # 34799 / BATCH_SIZE = 271
            end = offset + BATCH_SIZE
            y_train=sess.run(y_onehot)
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _, train_acc = sess.run([train_op, accuracy_op],
                                    feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5, LAMBDA: 1e-5})
            total_train_acc += (train_acc * len(batch_x))
        y_test = sess.run(y_onehot2)
        train_accuracy.append(total_train_acc / num_examples)
        test_acc = evaluate(X_test, y_test)
        test_accuracy.append(test_acc)
        print("Valid Accuracy = {:.3f}".format(test_acc))

    # saver.save(sess, 'model/alexnet.ckpt')
    # print("Model saved")