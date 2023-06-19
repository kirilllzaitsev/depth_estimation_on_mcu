import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

sys.path.append(
    "/media/master/wext/msc_studies/second_semester/microcontrollers/project/stm32/code/nyuv2_torch_ds_adapter.py"
)
import json
import os

import albumentations as A
import cv2
import torchvision.transforms as transforms
from config import cfg
from torch.utils.data import DataLoader, Dataset, Subset, random_split

# changes to orig dataset
# scale_size -> target_size


class BaseDataset(Dataset):
    def __init__(self, crop_size, fold_ratio=1, args=None, is_maxim=True):
        self.count = 0
        self.fold_ratio = fold_ratio
        self.is_maxim = is_maxim

        train_transform = [
            # A.HorizontalFlip(),
            A.RandomCrop(crop_size[1], crop_size[0]),
            # A.RandomBrightnessContrast(),
            # A.RandomGamma(),
            # A.HueSaturationValue(),
        ]
        test_transform = [
            A.CenterCrop(crop_size[1], crop_size[0]),
        ]
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.to_tensor = transforms.ToTensor()
        self.args = args

    def readTXT(self, txt_path):
        with open(txt_path, "r") as f:
            listInTXT = [line.strip() for line in f]

        return listInTXT

    def augment_training_data(self, image, depth):
        H, W, C = image.shape

        # if self.count % 4 == 0:
        #     alpha = random.random()
        #     beta = random.random()
        #     p = 0.75

        #     l = int(alpha * W)
        #     w = int(max((W - alpha * W) * beta * p, 1))

        #     image[:, l : l + w, 0] = depth[:, l : l + w]
        #     image[:, l : l + w, 1] = depth[:, l : l + w]
        #     image[:, l : l + w, 2] = depth[:, l : l + w]

        image, depth = self.common_augment(image, depth, self.train_transform)

        self.count += 1

        return image, depth

    def common_augment(self, image, depth, transform):
        additional_targets = {"depth": "mask"}
        aug = A.Compose(transforms=transform, additional_targets=additional_targets)
        augmented = aug(image=image, depth=depth)
        image = augmented["image"]
        depth = augmented["depth"]

        if self.is_maxim:
            image = self.apply_ai8x_transforms(image)
            depth = self.apply_ai8x_transforms(depth)
        # depth = depth.squeeze()
        return image, depth

    def apply_ai8x_transforms(self, x):
        import ai8x

        x = self.to_tensor(x)
        x = ai8x.normalize(self.args)(x)
        x = ai8x.fold(fold_ratio=self.fold_ratio)(x)
        return x

    def augment_test_data(self, image, depth):
        image, depth = self.common_augment(image, depth, self.test_transform)

        return image, depth


class nyudepthv2(BaseDataset):
    def __init__(
        self,
        data_path,
        args,
        filenames_path="/media/master/text/cv_data/nyuv2/nyu_data/data",
        is_train=True,
        crop_size=(448, 576),
        scale_size=None,
        fold_ratio=1,
    ):
        super().__init__(
            crop_size,
            fold_ratio=fold_ratio,
            args=args,
            is_maxim=getattr(args, "is_maxim", True),
        )

        # if crop_size[0] > 480:
        #     scale_size = (int(crop_size[0] * 640 / 480), crop_size[0])

        self.scale_size = scale_size

        self.is_train = is_train
        self.data_path = Path(data_path)

        self.image_path_list = []
        self.depth_path_list = []
        self.base_dir = Path(filenames_path).parent

        txt_path = Path(filenames_path)
        if is_train:
            txt_path /= "nyu2_train.csv"
            self.data_path = Path(self.data_path / "nyu2_train")
        else:
            txt_path /= "nyu2_test.csv"
            self.data_path = Path(self.data_path / "nyu2_test")

        import pandas as pd

        self.df = pd.read_csv(txt_path, header=None, names=["img_path", "depth_path"])
        phase = "train" if is_train else "test"
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.df)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = str(self.base_dir / self.df.loc[idx, "img_path"])
        gt_path = str(self.base_dir / self.df.loc[idx, "depth_path"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype("float32")

        # depth = depth / 1000.0  # convert in meters

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[1], self.scale_size[0]))
            depth = cv2.resize(depth, (self.scale_size[1], self.scale_size[0]))

        image = np.expand_dims(image, axis=2)
        depth = np.expand_dims(depth, axis=2)
        depth = depth.astype("float32")
        image = image.astype("float32")
        # depth = tf.image.convert_image_dtype(depth, tf.float32)

        # image = tf.image.convert_image_dtype(image, tf.float32)
        depth /= 1000.0  # convert in meters
        # min_depth, max_depth = 0.002, 4
        # depth = (depth - min_depth) / (max_depth - min_depth)
        depth = np.clip(depth, 0, 1)
        image /= 255.0

        return image, depth


def get_tf_nyuv2_ds(data_path, args):
    nyuv2_ds_train = nyudepthv2(
        data_path=data_path,
        filenames_path=data_path,
        args=args,
        is_train=True,
        crop_size=args.crop_size,
        scale_size=args.target_size,
        fold_ratio=args.out_fold_ratio,
    )
    nyuv2_ds_test = nyudepthv2(
        data_path=data_path,
        filenames_path=data_path,
        is_train=False,
        crop_size=args.crop_size,
        scale_size=args.target_size,
        fold_ratio=args.out_fold_ratio,
        args=args,
    )
    _ = nyuv2_ds_train[0]

    def generator(ds):
        # for images, labels in nyuv2_loader:
        for sample in ds:
            # Yield data batch-by-batch
            img, depth = sample
            # img /= 255.0
            yield (img, depth)
            # yield images.numpy(), labels.numpy()

    # Use output_signature to specify the output format and shapes
    output_signature = (
        tf.TensorSpec(shape=(*args.target_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(*args.target_size, 1), dtype=tf.float32),
    )

    val_size = int(0.2 * len(nyuv2_ds_train))  # 20% of the dataset

    # Split the dataset into training and validation sets
    seed_generator = torch.Generator().manual_seed(111)
    train_dataset, val_dataset = random_split(
        nyuv2_ds_train,
        [len(nyuv2_ds_train) - val_size, val_size],
        generator=seed_generator,
    )
    datasets = []

    print("Train size: ", len(train_dataset))
    print("Val size: ", len(val_dataset))
    print("Test size: ", len(nyuv2_ds_test))

    for ds in [train_dataset, val_dataset, nyuv2_ds_test]:
        if cfg.do_overfit:
            ds = Subset(ds, range(1))
        elif cfg.do_subsample:
            ds = Subset(ds, range(0, 3000))
        tf_dataset = tf.data.Dataset.from_generator(
            partial(generator, ds=ds), output_signature=output_signature
        ).batch(args.batch_size).prefetch(1)
        datasets.append(tf_dataset)
    return datasets
