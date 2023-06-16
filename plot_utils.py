import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    # Plot the training history
    plt.plot(history.history["mean_squared_error"], label="mean_squared_error")
    plt.plot(history.history["val_mean_squared_error"], label="val_mean_squared_error")
    plt.xlabel("Epoch")
    plt.ylabel("mean_squared_error")
    plt.ylim([0, 1])
    plt.legend(loc="lower right")


def plot_sample_nyuv2(x):
    if isinstance(x, tuple):
        if len(x[0].shape) == 4:
            img = x[0][0]
            depth = x[1][0]
        else:
            img = x[0]
            depth = x[1]
    else:
        img = x[0]
        depth = x[1]
    img, depth = img.numpy().squeeze(), depth.numpy().squeeze()
    print(img.shape, depth.shape)
    print(f'img: {img.min()}, {img.max()}')
    print(f'depth: {depth.min()}, {depth.max()}')
    plt.imshow(img / 255)
    plt.show()
    plt.imshow(depth)
    plt.show()
