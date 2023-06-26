import os
import random
import tempfile
import zipfile

import numpy as np
import tensorflow as tf


def print_meta():
    print("TensorFlow version: ", tf.__version__)
    print(
        "GPU is",
        "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE",
    )


def save_test_data(save_dir, test_images, test_labels):
    # save the test data as numpy arrays
    np.save(f"{save_dir}/x_test_depth.npy", test_images.numpy())
    np.save(f"{save_dir}/y_test_depth.npy", test_labels.numpy())

    # print the location of the files
    print(
        "Test image data location: ",
        os.path.abspath(f"{save_dir}/x_test_depth.npy"),
    )
    print(
        "Test labels location: ",
        os.path.abspath(f"{save_dir}/y_test_depth.npy"),
    )


def set_seed(seed=1234):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_gzipped_model_size(file):
    _, zipped_file = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zipped_file, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)
