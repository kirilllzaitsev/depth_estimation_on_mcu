import tensorflow as tf
from config import cfg


def calculate_loss(target, pred):
    # Edges
    if not isinstance(target, tf.Tensor):
        target = tf.convert_to_tensor(target)
    if not isinstance(pred, tf.Tensor):
        pred = tf.convert_to_tensor(pred)
    if len(target.shape) == 3:
        target = target[tf.newaxis, ...]
    if len(pred.shape) == 3:
        pred = pred[tf.newaxis, ...]
    dy_true, dx_true = tf.image.image_gradients(target)
    dy_pred, dx_pred = tf.image.image_gradients(pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    # Depth smoothness
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y

    depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
        abs(smoothness_y)
    )

    # Structural similarity (SSIM) index
    ssim_loss = tf.reduce_mean(
        1
        - tf.image.ssim(
            target, pred, max_val=cfg.w, filter_size=7, k1=0.01**2, k2=0.03**2
        )
    )
    # Point-wise depth
    l1_loss = tf.reduce_mean(tf.abs(target - pred))

    loss = (
        (cfg.ssim_loss_weight * ssim_loss)
        + (cfg.l1_loss_weight * l1_loss)
        + (cfg.edge_loss_weight * depth_smoothness_loss)
    )

    return loss
