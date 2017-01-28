from os import listdir
from random import shuffle
import time
import re
import tensorflow as tf
import numpy as np
import model as m
from loader import Loader


image_dir = 'input_images/'
label_dir = 'label_images/'

label_files = [filename for filename in listdir(label_dir)]

label_files.sort()
#shuffle(label_files)

test_label_files = label_files[:10]
train_label_files = label_files[10:]

test_loader = Loader(test_label_files, 5, 1, randomize = False)
train_loader = Loader(train_label_files, 5, 1, randomize = False)

test_loader.start()
train_loader.start()

test_batch = test_loader.get_batch(5)
test_loader.stop()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter('/tmp/dstl', graph=sess.graph)
  
  print(time.time(), 'starting')

  while True:
    batch = train_loader.get_batch(1)

    break
    filename = batch[0][0]
    input_image = batch[0][1]
    label_image = batch[0][2]

    pos_weight = 1 #(900*900*10 - np.count_nonzero(labels))/np.count_nonzero(labels)

    _, error, summary, step, learning_rate = sess.run([m.train, m.error, m.summary, m.global_step, m.learning_rate], feed_dict={ 
      m.input_image: [input_image],  
      m.label_image: [label_image], 
      m.pos_weight: pos_weight 
    })

    #summary_writer.add_summary(summary, step)

    print(time.time(), step, error, learning_rate, filename, pos_weight)



train_loader.stop()