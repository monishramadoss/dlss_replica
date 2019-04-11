import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import math
from model import dlss_autoencoder

TILESIZE = 8
SCALING = 2
BATCHSIZE = 32

tf.enable_eager_execution()

dataset_builder = tfds.builder("cifar10")
dataset_builder.download_and_prepare()
train = dataset_builder.as_dataset(split=tfds.Split.TRAIN)
test = dataset_builder.as_dataset(split=tfds.Split.TEST)
test_data = list()
train_data = list()


for x in test:
    test_data.append(x['image'])
for x in train:
    train_data.append(x['image'])

test = tf.extract_image_patches(test_data, ksizes=[1, TILESIZE, TILESIZE, 1], strides=[1, TILESIZE, TILESIZE, 1], rates=[1,1,1,1], padding='SAME')
test = tf.reshape(test, (-1, TILESIZE, TILESIZE, 3))
y_test = test
x_test = tf.image.resize(test, (TILESIZE//SCALING, TILESIZE//SCALING))


train = tf.extract_image_patches(train_data, ksizes=[1, TILESIZE, TILESIZE, 1], strides=[1, TILESIZE, TILESIZE, 1], rates=[1,1,1,1], padding='SAME')
train = tf.reshape(train, (-1, TILESIZE, TILESIZE, 3))
y = train
X = tf.image.resize(train, (TILESIZE//SCALING, TILESIZE//SCALING))



optimizer = tf.train.AdamOptimizer()
loss_history = list()
logits = list()
dlss = dlss_autoencoder(SCALING)
dlss.reset_hidden(X[0])

for i, image in enumerate(X):
    with tf.GradientTape() as tape:
        logit = dlss(image)
        loss_value = tf.losses.mean_squared_error(y[i], logit)
    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, dlss_autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, dlss_autoencoder.trainable_variables), global_step=tf.train.get_or_create_global_step())


