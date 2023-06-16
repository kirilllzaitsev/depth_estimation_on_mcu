import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from config import cfg
from data import test_images, test_labels, train_images, train_labels


def print_meta():
    print("TensorFlow version: ", tf.__version__)
    print(
        "GPU is",
        "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE",
    )


def save_test_data():
    # save the test data as numpy arrays
    np.save(f"{cfg.save_test_data_dir}/x_test_depth.npy", test_images.astype(np.uint8))
    np.save(f"{cfg.save_test_data_dir}/y_test_depth.npy", test_labels.astype(np.uint8))
    # plot the first 5 images in the test set with their labels
    # map class labels to names
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(test_images.astype(np.uint8)[i], cmap="gray")
        plt.title("Label: %s" % class_names[test_labels[i]])

    # print the location of the files
    print(
        "Test image data location: ",
        os.path.abspath(f"{cfg.save_test_data_dir}/x_test_depth.npy"),
    )
    print(
        "Test labels location: ",
        os.path.abspath(f"{cfg.save_test_data_dir}/y_test_depth.npy"),
    )
