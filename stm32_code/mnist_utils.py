import matplotlib.pyplot as plt
import numpy as np
from config import cfg


def plot_samples(train_labels, train_images_float):
    samples_per_class = 2
    for y, cls in enumerate(cfg.classes):
        idxs = np.flatnonzero(train_labels == y)
        if len(idxs) < samples_per_class:
            print(f"Warning: class {cls} has only {len(idxs)} samples. skipping")
            continue
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * cfg.num_classes + y + 1
            plt.subplot(samples_per_class, cfg.num_classes, plt_idx)
            plt.imshow(train_images_float[idx])
            plt.axis("off")
            if i == 0:
                plt.title(cls)
