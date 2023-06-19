import matplotlib.pyplot as plt
import numpy as np
from loss import calculate_loss


def plot_history(history, save_path=None):
    # Plot the training history
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
