from os import listdir
from random import shuffle, randint
import time

import numpy as np
import tensorflow as tf
import tifffile as tiff
from scipy import ndimage, misc

from loader import Loader
import model_bn as m


label_files = [filename for filename in listdir('input_images')]
label_files.sort()
#shuffle(label_files)


def data_processor(filename, config):
  input_image = tiff.imread('input_images/{0}'.format(filename))
  label_image = tiff.imread('label_images/{0}'.format(filename[:-1]))
  label_image = label_image[:, :, :1] / 255

  if config.augment:
    angle = randint(0, 360)
    input_image = ndimage.rotate(input_image, angle, reshape=False)
    label_image = ndimage.rotate(label_image, angle, reshape=False)

  input_image = (input_image - np.mean(input_image)) / np.std(input_image)
  return (filename, input_image, label_image)


train_label_files = label_files[5:]
train_loader = Loader(train_label_files, 5, 1, processor=data_processor)
train_loader.start()


test_label_files = label_files[:5]
test_loader = Loader(test_label_files, 5, 1, processor=data_processor, randomize=False, augment=False)
test_loader.start()
test_batch = test_loader.get_batch(5)
test_loader.stop()



shouldLoad = False
modelName = "{0}-0".format(m.modelName)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
  saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
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

    #if step % 20 == 0:
    #   save_path = saver.save(sess, modelName)
    
    summary_writer.add_summary(summary, step)

    if step % 100 == 0:
      ave_error = 0
      for item in test_batch:
        filename = item[0]
        input_image = item[1]
        label_image = item[2]

        test_error, learning_rate, result = sess.run([m.error, m.learning_rate, m.result], feed_dict={ 
          m.input_image: [input_image],  
          m.label_image: [label_image],
          m.is_train: False
        })

        ave_error += test_error

        result_image = result[0].reshape((1500, 1500))
        misc.imsave("training/{0}-{1}-{2}.png".format(filename, step, modelName), result_image)

      print(time.time(), step, error, ave_error / 5.0, learning_rate)


train_loader.stop()