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
