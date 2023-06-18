import os

import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# import tensorflow.keras as keras
from keras import layers


def get_model(
    img_size,
    num_classes,
    in_channels=3,
    use_qat=False,
    use_pruning=False,
    use_pruning_struct=False,
    use_dynamic_sparsity=False,
    pruned_model_unstructured_for_export=None,
):
    # inputs = keras.Input(shape=img_size + (in_channels,))
    inputs = layers.Input(shape=(*img_size, in_channels), name="input")

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    filters = [16//2*3, 32//2*3, 64//2*3]
    # filters = [16, 32, 64]
    # filters = [16, 32, 64, 128, 256]
    x = layers.Conv2D(filters[0], in_channels, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.relu)(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filter in filters[1:]:
        x = layers.Activation(tf.nn.relu)(x)
        x = layers.SeparableConv2D(filter, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation(tf.nn.relu)(x)
        x = layers.SeparableConv2D(filter, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filter, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filter in filters[::-1]:
        x = layers.Activation(tf.nn.relu)(x)
        x = layers.Conv2DTranspose(filter, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation(tf.nn.relu)(x)
        x = layers.Conv2DTranspose(filter, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filter, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # x = layers.Conv2D(
    #     1, in_channels, activation=None, padding="same"
    # )(x)
    # outputs = layers.Activation("sigmoid", name="output")(x)

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(
        # 1, in_channels, activation="tanh", padding="same", name="output"
        1,
        in_channels,
        activation="tanh",
        # activation="sigmoid",
        padding="same",
        name="output",
    )(x)

    # Define the model
    model = keras.Model(inputs, outputs)

    if use_qat and use_pruning:
        # PQAT
        quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
            pruned_model_unstructured_for_export
        )

        model = tfmot.quantization.keras.quantize_apply(
            quant_aware_annotate_model,
            # tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme(),
        )
    elif use_qat:
        # Convert the model to a quantization aware model
        model = tfmot.quantization.keras.quantize_model(model)
    elif use_pruning:
        if use_pruning_struct:
            # Strucutred pruning with constant sparsity
            pruning_params = {
                "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                    0.5, begin_step=2000, frequency=100
                ),
                "block_size": (1, 1),
            }
        else:
            if use_dynamic_sparsity:
                # Unstructured pruning with dynamic sparsity
                pruning_params = {
                    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.50,
                        final_sparsity=0.80,
                        begin_step=2000,
                        end_step=4000,
                        frequency=100,
                    )
                }
            else:
                pruning_params = {
                    "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                        0.5, begin_step=2000, frequency=100
                    ),
                }
        model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    return model


def save_pruned_model(pruned_model, pruned_keras_file):
    pruned_model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
    tf.keras.models.save_model(
        pruned_model_for_export, pruned_keras_file, include_optimizer=False
    )
    print("Saved pruned Keras model to:", os.path.abspath(pruned_keras_file))
    return pruned_model_for_export
