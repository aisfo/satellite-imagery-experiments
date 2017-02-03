from os import listdir
from random import shuffle
import time
import re
import tensorflow as tf
import numpy as np
import model as m
from loader import Loader
import scipy.misc


image_dir = 'input_images/'
label_files = [filename for filename in listdir(image_dir)]
label_files.sort()
shuffle(label_files)

test_label_files = label_files[:5]
train_label_files = label_files[5:]

test_loader = Loader(test_label_files, 5, 1, randomize = False, augment = False)
train_loader = Loader(train_label_files, 5, 1)

test_loader.start()
train_loader.start()

test_batch = test_loader.get_batch(5)
test_loader.stop()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

shouldLoad = False
modelName = 'msi-base'

saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)


with tf.Session(config=config) as sess:
  
  summary_writer = tf.summary.FileWriter('/home/aisfo/tmp/' + modelName, graph=sess.graph)

  if shouldLoad:
    saver.restore(sess, modelName)
  else:
    sess.run(tf.global_variables_initializer())
  
  print(time.time(), 'starting')

  for epoch in range(1000):
    batch = train_loader.get_batch(1)

    filename = batch[0][0]
    input_image = batch[0][1]
    label_image = batch[0][2]

    _, error, summary, step, learning_rate = sess.run([m.train, m.error, m.summary, m.global_step, m.learning_rate], feed_dict={ 
      m.input_image: [input_image],  
      m.label_image: [label_image]
    })

    save_path = saver.save(sess, modelName)
    summary_writer.add_summary(summary, step)

    if epoch % 100 == 0:
      for item in test_batch:
        filename = item[0]
        input_image = item[1]
        label_image = item[2]

        test_error, learning_rate, result = sess.run([m.error, m.learning_rate, m.test], feed_dict={ 
          m.input_image: [input_image],  
          m.label_image: [label_image],
          m.is_train: False
        })

        print(time.time(), step, error, test_error, learning_rate, filename)

        result_image = result[0].reshape((1500, 1500))
        scipy.misc.imsave("training/{0}-{1}-{2}.png".format(modelName, filename, step), result_image)





train_loader.stop()