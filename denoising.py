# Denoising of disturbed data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.utils import shuffle
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
Trainingsbilder = []

print("loading dataset:X_stego ")
name=[3]
for i in name:
    n = str(i)
    Pfad = "../dataset/LSB_MNIST_37/" + n
    # label=i
    for Datei in os.listdir(Pfad):
        img = os.path.join(Pfad,Datei)
        #print(img)
        #img = load_img(img,target_size=(28,28,1))
        img = load_img(img,grayscale=True,target_size=(28,28))
        img = img_to_array(img,  dtype=np.float32)
        img=img.reshape(1,28,28,1)
        Trainingsbilder.append(img)

#Convert list to tensor
Trainingsbilder = np.asarray([Trainingsbilder])
Trainingsbilder = Trainingsbilder.reshape(-1, 784)
#0-1
Trainingsbilder = Trainingsbilder/255
# print(Trainingsbilder)
X_train = np.asarray(Trainingsbilder, dtype = "float32")
print(X_train.shape)
print(X_train)
Trainingsbilder3= []
print("loading dataset:X_stego2")
name=[3]
for i in name:
    n = str(i)
    Pfad = "../dataset/LSB_MNIST_37(2)/" + n
    # label=i
    for Datei in os.listdir(Pfad):
        img = os.path.join(Pfad,Datei)
        #print(img)
        # img = load_img(img,target_size=(28,28,1))
        img = load_img(img,grayscale=True,target_size=(28,28))
        img = img_to_array(img,  dtype=np.float32)

        img=img.reshape(1,28,28,1)
        Trainingsbilder3.append(img)

#Convert list to tensor
Trainingsbilder3 = np.asarray([Trainingsbilder3])
Trainingsbilder3 = Trainingsbilder3.reshape(-1, 784)
#0-1
Trainingsbilder3 = Trainingsbilder3/255
# print(Trainingsbilder)
X_train3 = np.asarray(Trainingsbilder3, dtype = "float32")
print(X_train3.shape)
print(X_train3)
Trainingsbilder2 = []
print("loading dataset:X_pert")
name=[3]
for i in name:
    n = str(i)
    Pfad = "../dataset/MNIST/deal_M/train/" + n
    # label=i
    for Datei in os.listdir(Pfad):
        img = os.path.join(Pfad,Datei)
        #print(img)
        img = load_img(img,grayscale=True)
        img = img_to_array(img,  dtype=np.float32)
        #img=img.reshape(1,28,28,1)##
        img = img.reshape(1, 32, 32, 1)##
        Trainingsbilder2.append(img)
Trainingsbilder2 = np.asarray([Trainingsbilder2])
#Trainingsbilder2 = Trainingsbilder2.reshape(-1, 784)##
Trainingsbilder2 = Trainingsbilder2.reshape(6131, 32,32,1)##
Trainingsbilder2 = Trainingsbilder2/255
X_train2 = np.asarray(Trainingsbilder2, dtype = "float32")


Testbilder = []
print("loading dataset_test")
name=[3]
for i in name:
    n = str(i)
    Pfad = "../dataset/LSB_MNIST_37-test/" + n
    for Datei in os.listdir(Pfad):
        img = os.path.join(Pfad,Datei)
        #print(img)
        img = load_img(img,grayscale=True)
        img = img.resize((28, 28)) ##
        img = img_to_array(img,  dtype=np.float32)
        #img=img.reshape(1,28,28,1)##
        Testbilder.append(img)
#Convert lists to tensors
Testbilder = np.asarray([Testbilder])
#Testbilder = Testbilder.reshape(-1,784)##
Testbilder = Testbilder.reshape(6131,784)##
#0-1
Testbilder = Testbilder/255
X_test = np.asarray(Testbilder, dtype = "float32")
print(X_test.shape)
# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 64
# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)
X = tf.placeholder(tf.float32, shape=(None, 784))
X_pert = tf.placeholder(tf.float32, shape=(None, 784))
weights = {
    'encoder_w1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
    'encoder_w2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'decoder_w1': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], stddev=0.1)),
    'decoder_w2': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev=0.1))
}
bias = {
    "encoder_b1": tf.Variable(tf.truncated_normal([1, n_hidden_1], stddev=0.1)),
    "encoder_b2": tf.Variable(tf.truncated_normal([1, n_hidden_2], stddev=0.1)),
    "decoder_b1": tf.Variable(tf.truncated_normal([1, n_hidden_1], stddev=0.1)),
    "decoder_b2": tf.Variable(tf.truncated_normal([1, n_input], stddev=0.1))
}
def encoder(X):
    layer1 = tf.nn.sigmoid(tf.matmul(X, weights['encoder_w1']) + bias['encoder_b1'])
    print(layer1.shape)
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, weights['encoder_w2']) + bias['encoder_b2'])
    print(layer2.shape)
    return layer2
def decoder(x):
    layer1 = tf.nn.sigmoid(tf.matmul(x, weights['decoder_w1']) + bias['decoder_b1'])
    print(layer1.shape)
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, weights['decoder_w2']) + bias['decoder_b2'])
    print(layer2.shape)
    return layer2
encoder_op = encoder(X_pert)
decoder_op = decoder(encoder_op)

pred = decoder_op
print(pred.shape)
print('////')
entropy = tf.losses.log_loss(labels=X, predictions=decoder_op)
assert isinstance(entropy, object)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
num_batches = int(100/batch_size)

for epoch in range(5000):
    total_loss = 0
    for i in range(num_batches):
        example=X_train[i*batch_size:(i+1)*batch_size]  #stego
        example2=X_train2[i*batch_size:(i+1)*batch_size]  #pert
        example=example.reshape(64,784)##
        example2=example.reshape(64,784)##
        _, l = sess.run([optimizer, loss], feed_dict={X: example, X_pert: example2})
        total_loss += l
    print("Epoch {0}: {1}".format(epoch, total_loss))

X_train3=X_train3.reshape(6131,784)##

pred_img = sess.run(pred, feed_dict = {X:X_train3,X_pert:X_test})
print('pred_image.shape:',pred_img.shape)
pred_1=pred_img.reshape(-1,28,28,1)##

print(pred_1.shape)
f, a = plt.subplots(3, 10, figsize=(10, 3))
plt.axis('off')
for i in range(5):
    cv2.imwrite("../dataset/MNIST/img3-gan/%d.jpg" % i, pred_1[i]*255)
    a[0][i].imshow(np.reshape(X_test[i], (28, 28)))

    # a[1][i].imshow(np.reshape(x_noise[i], (28, ))
    a[1][i].imshow(np.reshape(pred_1[i], (28,28)))
plt.show()