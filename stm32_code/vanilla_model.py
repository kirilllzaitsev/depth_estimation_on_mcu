import keras
import tensorflow as tf
from config import cfg
from model import get_model


def fit(ds_train, epochs, img_size, ds_val=None):
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()
    loss = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.MeanSquaredError()
    # Build model
    fp_model = get_model(img_size, cfg.num_classes, in_channels=cfg.in_channels)
    # Compile the model
    fp_model.compile(optimizer="adam", loss=loss, metrics=metrics)

    es = tf.keras.callbacks.EarlyStopping(
        patience=cfg.es_patience, restore_best_weights=True
    )
    history = fp_model.fit(
        x=ds_train,
        epochs=epochs,
        validation_data=ds_val,
        callbacks=[es],
    )
    return fp_model, history