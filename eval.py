import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from config import cfg
from mnist_data import test_images, test_labels, train_images, train_labels


# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((test_image_indices.stop, *test_images[0].shape), dtype=int)
    for i, test_image_index in enumerate(test_image_indices):
        test_image = test_images[test_image_index]
        test_label = test_images[test_image_index]

        if test_image_index % 1000 == 0 and test_image_index > 0:
            print("Evaluated on %d images." % test_image_index)

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details["dtype"] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        predictions[i] = output

    return predictions


def eval_quantized_model(model_path, model_type):
    tflite_model_quant_int8_file = pathlib.Path(model_path)

    test_image_indices = range(test_images.shape[0])
    predictions = run_tflite_model(tflite_model_quant_int8_file, test_image_indices)

    mse = np.mean((test_images - predictions) ** 2)

    print(f"{model_type} model mse is {mse} (Number of test samples=20)")
