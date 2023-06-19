# set global seeds for reproducibility
import logging
import os
import random

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

dotenv.load_dotenv()
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
    # img_size = (64, 64)
    img_size = (32, 32)
    # img_size = (640 // 4, 480 // 4)
    num_classes = len(classes)
    classes = classes
    in_channels = 1


class modelcfg:
    ...


class traincfg:
    epochs = 50
    es_patience = 5

    ssim_loss_weight = 0.85
    l1_loss_weight = 0.1
    edge_loss_weight = 0.9


class metacfg:
    # do_overfit = True
    do_overfit = False
    do_shuffle = True
    do_subsample = True
    batch_size = 32
    take_first_n = 30

    is_cluster = os.path.exists("/cluster")

    save_model_dir = ""
    save_cfiles_dir = ""
    save_test_data_dir = ""
    base_kitti_dataset_dir = ""

    tmpdir = os.getenv("TMPDIR")
    logdir = ""
    path_to_project_dir = os.environ["path_to_project_dir"]
    ckpt_dir = f"{path_to_project_dir}/code/ckpt"


class cfg(datacfg, modelcfg, traincfg, metacfg):
    ...


if cfg.is_cluster:
    if not os.path.exists(f"{cfg.tmpdir}/cluster"):
        os.system(
            f"tar -xvf /cluster/project/rsl/kzaitsev/nyuv2.tar.gz -C {cfg.tmpdir} > /dev/null 2>&1"
        )

metacfg.save_model_dir = f"{metacfg.path_to_project_dir}/code/models"
metacfg.save_cfiles_dir = f"{metacfg.path_to_project_dir}/code/cfiles"
metacfg.save_test_data_dir = f"{metacfg.path_to_project_dir}/code/test_data"

if metacfg.is_cluster:
    metacfg.base_kitti_dataset_dir = os.path.join(
        metacfg.tmpdir, os.environ["base_kitti_dataset_dir"]
    )
    metacfg.logdir = "/cluster/scratch/kzaitse/stm32/logs"
else:
    metacfg.base_kitti_dataset_dir = os.environ["base_kitti_dataset_dir"]
    metacfg.logdir = "/tmp"
