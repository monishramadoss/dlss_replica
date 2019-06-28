import numpy as np
from itertools import product
from multiprocessing import Pool
import tensorflow_datasets as tfds
import tensorflow as tf
import os

def proc(buf):
    config = tf.compat.v1.ConfigProto(device_count={'GPU':0})
    sess =  tf.compat.v1.Session(config=config)
    with sess.as_default():
        return buf['image'].eval()


def preProcess(data):   

    if (not os.path.isfile('train.npy') or not os.path.isfile('test.npy')):
        tr_buf = [x for x in data['trainB']]
        te_buf = [x for x in data['testB']]

        shape = tr_buf[0]['image'].shape
    
        train = np.zeros([len(tr_buf), shape[0], shape[1], shape[2]])
        test = np.zeros([len(te_buf), shape[0], shape[1], shape[2]])    

        with Pool(5) as p:
            tr_buf = p.map(proc, tr_buf)
            train = np.array(tr_buf)
            np.save('train.npy', train)

            te_buf = p.map(proc, te_buf)
            test = np.array(te_buf)
            np.save('test.npy', test)
    else:
        train = np.load('train.npy')
        test = np.load('test.npy')

    return (train, test)
        