import argparse

import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import tensorflow as tf
import tensorflow.keras as keras
from c_utils import write_model_h
from config import cfg
from converters import (
    check_quantized_model,
    dynamic_range_quantization,
    eight_bit_quantization,
    keras_to_tflite,
)
from eval import eval_quantized_model, run_tflite_model
from loss import calculate_loss
from metrics import calc_metrics
from mnist_utils import plot_samples
from model import get_model
from nyuv2_torch_ds_adapter import get_tf_nyuv2_ds
from utils import save_test_data, set_seed
from vanilla_model import fit as fit_vanilla

set_seed()
args = argparse.Namespace()
args.truncate_testset = False
# args.target_size = (64, 64)
args.crop_size = (640, 480)
# args.target_size = (64, 64)
args.target_size = cfg.img_size
args.out_fold_ratio = 1
args.is_maxim = False
ds_train, ds_val, ds_test = get_tf_nyuv2_ds(cfg.base_kitti_dataset_dir, args)
# x=next(iter(ds_train.shuffle(100).batch(2).take(1)))
# pu.plot_sample_nyuv2(x)
train_size = 20
ds_train = ds_train.batch(2).take(train_size)
test_size = 20
ds_val = ds_val.batch(2).take(test_size)


keras.backend.clear_session()
# loss = tf.keras.losses.MeanSquaredError()
metrics = tf.keras.metrics.Mean(name="loss")


# Define a custom metric
def custom_metric(y_true, y_pred, sample_weight=None):
    metric_value = calculate_loss(y_true, y_pred)
    metrics.update_state(metric_value, sample_weight=sample_weight)
    return metric_value


# Build model
model = get_model(
    cfg.img_size, cfg.num_classes, in_channels=cfg.in_channels, use_qat=False
)
# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=calculate_loss, metrics=[custom_metric])

es = tf.keras.callbacks.EarlyStopping(
    patience=cfg.es_patience, restore_best_weights=True
)
history = model.fit(
    x=ds_train,
    epochs=cfg.epochs * 3 * 2,
    validation_data=ds_val,
    # callbacks=[es],
    verbose=1,
)


def plot_eval_results(pred_depth, true_depth, rgb, save_path=None):
    mae = np.mean(np.abs(pred_depth - true_depth))
    rmse = np.mean(np.square(pred_depth - true_depth))
    print(
        f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, loss: {calculate_loss(pred_depth, true_depth):.2f}"
    )
    plt.figure(figsize=(5, 3))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # plt.subplot(1, 2, 2)
    fix, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(pred_depth[..., 0], cmap="gray")
    axs[0].set_title("Predicted depth")
    axs[1].imshow(true_depth[..., 0], cmap="gray")
    axs[1].set_title("True depth")
    axs[2].imshow(rgb)
    axs[2].set_title("RGB")
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)


x_val = next(iter(ds_val))
x_train = next(iter(ds_train))
out = model.predict(ds_train)
plot_eval_results(out[0], x_train[1][0], x_train[0][0], save_path="vanilla_results.png")
