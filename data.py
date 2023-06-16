import cv2
import numpy as np
import tensorflow as tf
from config import cfg

# Load Fashion-Mnist dataset, we can use Tensorflow for this
(train_images, train_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = (
    train_images[:100],
    train_labels[:100],
), (test_images[:100], test_labels[:100])


def transform_imgs(imgs):
    new_imgs = []
    for i, img in enumerate(imgs):
        new_imgs.append(cv2.resize(img, cfg.img_size, interpolation=cv2.INTER_NEAREST))
    return np.array(new_imgs)


train_images = transform_imgs(train_images)
test_images = transform_imgs(test_images)

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0


train_images = (
    train_images[..., tf.newaxis] if len(train_images.shape) == 3 else train_images
)
test_images = (
    test_images[..., tf.newaxis] if len(test_images.shape) == 3 else test_images
)

if cfg.do_overfit:
    train_images = train_images[: cfg.take_first_n]
    train_labels = train_labels[: cfg.take_first_n]
    test_images = test_images[: cfg.take_first_n]
    test_labels = test_labels[: cfg.take_first_n]

print("Shape of train dataset: {}".format(train_images.shape))
print("Shape of train labels: {}".format(train_labels.shape))
print("Shape of test dataset: {}".format(test_images.shape))
print("Shape of test labels: {}".format(test_labels.shape))
