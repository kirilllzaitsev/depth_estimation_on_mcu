import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from loss import calculate_loss


def plot_history(history, save_path=None):
    if not isinstance(history, dict):
        history = history.history
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.ylim([0, 1])
    plt.legend(loc="upper right")
    if save_path is not None:
        plt.savefig(save_path)


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
    print(f"img: {img.min()}, {img.max()}")
    print(f"depth: {depth.min()}, {depth.max()}")
    plt.imshow(img / 255)
    plt.show()
    plt.imshow(depth)
    plt.show()


def plot_eval_results(pred_depth, true_depth, rgb, history=None):
    mae = np.mean(np.abs(pred_depth - true_depth))
    rmse = np.mean(np.square(pred_depth - true_depth))
    print(
        f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, loss: {calculate_loss(pred_depth, true_depth):.2f}"
    )
    if history is not None:
        plt.figure(figsize=(5, 3))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

    fix, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(pred_depth[..., 0], cmap="gray")
    axs[0].set_title("Predicted depth")
    axs[1].imshow(true_depth[..., 0], cmap="gray")
    axs[1].set_title("True depth")
    axs[2].imshow(rgb)
    axs[2].set_title("RGB")
    plt.show()


def plot_weight_distribution(model):
    num_layers = len(model.layers)
    layers_per_row = 5
    fig, axs = plt.subplots(
        num_layers // layers_per_row + 1, layers_per_row, figsize=(20, 20)
    )
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        row = idx // layers_per_row
        row = (
            num_layers // layers_per_row if row > num_layers // layers_per_row else row
        )
        col = idx % layers_per_row
        if len(layer.get_weights()) == 0:
            # print(f"Skipping layer {layer.name} as it has no weights")
            continue
        axs[row, col].hist(layer.get_weights()[0].flatten(), bins=10)
        axs[row, col].set_title(layer.name)
        axs[row, col].set_yscale("log")
        axs[row, col].set_xticks([-1, 0, 1])
    plt.tight_layout()
    plt.show()
