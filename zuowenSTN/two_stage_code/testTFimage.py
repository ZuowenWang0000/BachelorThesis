import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('cat.jpg')
implot = plt.imshow(img)
plt.show()

sess = tf.Session()
print("start transformation!")
trans_x = np.array([20])
trans_y = np.array([40])
rot = np.array([0.4])

img = img.reshape(1, 420,750,3)
print(img.shape)

ones = tf.ones(shape=tf.shape(trans_x))
zeros = tf.zeros(shape=tf.shape(trans_x))

trans = tf.stack([ones, zeros, -trans_x,
                      zeros, ones, -trans_y,
                      zeros, zeros], axis=1)

x = tf.contrib.image.transform(img, trans, interpolation='BILINEAR')
x = tf.contrib.image.rotate(x, rot, interpolation='BILINEAR')

trans = tf.stack([ones, zeros, trans_x,
                      zeros, ones, trans_y,
                      zeros, zeros], axis=1)

x = tf.contrib.image.rotate(x, -rot, interpolation='BILINEAR')
x = tf.contrib.image.transform(x, trans, interpolation='BILINEAR')


img_new = sess.run(x)
img_new = img_new.reshape(420, 750, 3).astype(dtype=int)
plt.imshow(img_new)
plt.show()