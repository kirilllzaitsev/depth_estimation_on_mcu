# set global seeds for reproducibility
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.random.set_seed(1234)
np.random.seed(1234)
random.seed(1234)

# Setting parameters for plotting
plt.rcParams["figure.figsize"] = (15.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

logging.getLogger("tensorflow").setLevel(logging.DEBUG)


classes = [
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


class datacfg:
    h, w = 64, 64
    img_size = (64, 64)
    num_classes = len(classes)
    classes = classes
    in_channels = 3


class modelcfg:
    ...


class traincfg:
    epochs = 30
    es_patience = 5

    ssim_loss_weight = 0.85
    l1_loss_weight = 0.1
    edge_loss_weight = 0.9


class metacfg:
    do_overfit = True
    take_first_n = 30

    save_model_dir = "/media/master/wext/msc_studies/second_semester/microcontrollers/project/stm32/code/models"
    save_cfiles_dir = "/media/master/wext/msc_studies/second_semester/microcontrollers/project/stm32/code/cfiles"
    save_test_data_dir = "/media/master/wext/msc_studies/second_semester/microcontrollers/project/stm32/code/test_data"


class cfg(datacfg, modelcfg, traincfg, metacfg):
    ...
