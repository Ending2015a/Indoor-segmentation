from __future__ import print_function

import argparse
import os
import sys
import time
import scipy.io as sio
import cv2

import tensorflow as tf
import numpy as np

from model import DeepLabResNetModel

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

NUM_CLASSES = 150
SAVE_DIR = './output/'
INPUT_DIR = './input/'
RESTORE_PATH = './restore_weights/'
matfn = 'color150.mat'

flags = tf.app.flags

flags.DEFINE_string('name', 'server', 'ID which will be used in log file')
flags.DEFINE_integer('port', 8888, 'Server socket port')
flags.DEFINE_string('logfile', './server.log', 'Log file')
flags.DEFINE_integer('width', 640, 'Width of input image')
flags.DEFINE_integer('height', 320, 'Height of input image')
flags.DEFINE_integer('device', 0, 'GPU device')

FLAGS = flags.FLAGS

def read_labelcolours(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    shape = color_table.shape
    color_list = [tuple(color_table[i]) for i in range(shape[0])]

    return color_list

def decode_labels(mask, num_classes=150):
    label_colours = read_labelcolours(matfn)

    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    output = np.zeros((h, w, 3), dtype=np.uint8)

    h, w, c = mask.shape

    for i in range(h):
        for j in range(w):
            k = mask[i,j,1]
            if k < num_classes:
                output[i, j, 0:2] = label_colours[k, 2:0]

    return output



def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

    

def main(argv=None):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)

    input_img = tf.placeholder(tf.float32, [1, FLAGS.height, FLAGS.width, 3])


    # Create network.
    net = DeepLabResNetModel({'data': input_img}, is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc_out']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    ckpt = tf.train.get_checkpoint_state(RESTORE_PATH)

    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
        load_step = 0


    # create server
    server = serverSOck(name=FLAGS.name)
    server.create(port=FLAGS.port)

    server.waitForClient()

    while True:
        print('wait for task')
        img = server.recv()
        img = cv2.imdecode('.jpg', img)
        h, w, c = img.shape
        img = cv2.resize(img, (FLAGS.width, FLAGS.height)).astype(float) - IMG_MEAN

        img = np.expand_dims(img, axis=0)

        preds = sess.run(pred, feed_dict={input_img: img})

        msk = decode_labels(preds, numclasses=NUM_CLASSES)
        
        msk = cv2.resize(msk, (w, h))

        msk = cv2.imencode('.jpg', msk)

        server.send(msk)

    server.close()

if __name__ == '__main__':
    tf.app.run()
