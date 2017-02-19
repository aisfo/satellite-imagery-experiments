
from os import listdir
from random import shuffle, randint
import time

import numpy as np
import tensorflow as tf
import tifffile as tiff
from scipy import ndimage, misc

from loader import Loader
import model_residual as m


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


test_label_files = label_files[:5]
test_loader = Loader(test_label_files, 5, processor=data_processor)
test_loader.start()
test_batch = test_loader.get_batch(5)
test_loader.stop()

batch_size = 2
train_label_files = label_files[5:]
train_loader = Loader(train_label_files, batch_size * 4, processor=data_processor, randomize=True, augment=True)
train_loader.start()


shouldLoad = False
modelName = m.modelName + "-x2-msr"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True

with tf.Session(config=config) as sess:
  saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
  summary_writer = tf.summary.FileWriter('summary/{0}'.format(modelName), graph=sess.graph)
  summary_writer_test = tf.summary.FileWriter('summary/{0}-test'.format(modelName), graph=sess.graph)

  if shouldLoad:
    saver.restore(sess, modelName)
  else:
    sess.run(tf.global_variables_initializer())
  
  startTime = time.time()
  print("{0}: training start;".format(startTime))

  while True:
    batch = train_loader.get_batch(batch_size)
    filenames = batch[0]
    input_images = batch[1]
    label_images = batch[2]

    _, error, summary, step = sess.run([m.train, m.error, m.summary, m.global_step], feed_dict={ 
      m.input_image: input_images,  
      m.label_image: label_images
    })

    #if step % 20 == 0:
    #   save_path = saver.save(sess, modelName)
    summary_writer.add_summary(summary, step)
    #print(time.time() - startTime, step, error)
    if step % (100 / batch_size) == 0:
      filenames = test_batch[0]
      input_images = test_batch[1]
      label_images = test_batch[2]

      test_error, learning_rate, result, summary = sess.run([m.error, m.learning_rate, m.result, m.test_summary], feed_dict={ 
        m.input_image: input_images,  
        m.label_image: label_images,
        m.is_train: False
      })

      summary_writer_test.add_summary(summary, step)

      print("{0}: step {1}; train {2}; test {3}; lrate {4};".format(time.time() - startTime, step, error, test_error, learning_rate))

      filename = filenames[0]
      result_image = result[0].reshape((1500, 1500))
      misc.imsave("test_results/{0}-{1}-{2}.png".format(modelName, filename, step), result_image)


    if step == 12000: break

train_loader.stop()
