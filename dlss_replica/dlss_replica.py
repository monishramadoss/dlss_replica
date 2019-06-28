import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import datetime
import math
from model import dlss_autoencoder
import utils
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

EPOCHS = 100
TILESIZE = 256
SCALING = 2
BATCHSIZE = 32


if(__name__ == '__main__'):
    
    tf.compat.v1.enable_eager_execution()

    dataset_builder = tfds.image.CycleGAN()
    dataset_builder.download_and_prepare()
    data = dataset_builder.as_dataset()

    train, test = utils.preProcess(data)

    train = tf.convert_to_tensor(train, dtype=tf.float32)
    test = tf.convert_to_tensor(test, dtype=tf.float32)

    y = train
    X = tf.image.resize(train, (TILESIZE//SCALING, TILESIZE//SCALING))


    dlss = dlss_autoencoder(SCALING)
    dlss.reset_hidden(y[0])

    dlss.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    dlss.fit(X, y, batch_size=BATCHSIZE, epochs=EPOCHS, callbacks=[tensorboard_callback])
 