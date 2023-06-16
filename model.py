import tensorflow.keras as keras
from tensorflow.keras import layers


def get_model(img_size, num_classes, in_channels=1, use_qas=False):
    # inputs = keras.Input(shape=img_size + (in_channels,))
    inputs = layers.Input(shape=(*img_size, in_channels), name="input")

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    filters = [16, 32]
    x = layers.Conv2D(filters[0], in_channels, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filter in filters[1:]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filter, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
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
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filter, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filter, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filter, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(
        1, in_channels, activation=None, padding="same", name="output"
    )(x)

    # Define the model
    model = keras.Model(inputs, outputs)

    if use_qas:
        import tensorflow_model_optimization as tfmot

        # Convert the model to a quantization aware model
        model = tfmot.quantization.keras.quantize_model(model)

    return model
