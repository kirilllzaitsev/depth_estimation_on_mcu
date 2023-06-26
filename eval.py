import json
import pathlib

import numpy as np
import tensorflow as tf


# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_ds):
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    maes = []
    mses = []
    sample = next(iter(test_ds))
    if len(sample[0].shape) == 3:
        ds = test_ds
    else:
        ds = test_ds.unbatch().batch(1)
    for i, (test_image, test_depth) in enumerate(ds):
        if i % 100 == 0 and i > 0:
            print("Evaluated on %d images." % i)

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details["dtype"] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        # test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        if len(test_image.shape) == 3:
            test_image = np.expand_dims(test_image, axis=0)
        if not isinstance(test_image, np.ndarray):
            test_image = test_image.numpy()
        test_image = test_image.astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        mse = np.mean((test_depth - output) ** 2)
        mae = np.mean(np.abs(test_depth - output))
        maes.append(mae)
        mses.append(mse)
        predictions.append(output)
    metrics = {
        "mse": np.mean(mses),
        "mae": np.mean(maes),
    }
    return predictions, metrics


def eval_quantized_model_in_tflite(tflite_path, test_ds):
    tflite_model_quant_int8_file = pathlib.Path(tflite_path)

    predictions, metrics = run_tflite_model(tflite_model_quant_int8_file, test_ds)

    return metrics


def eval_model(
    test_ds, model_name, model=None, tflite_path=None, metrics_file_path=None
):
    metrics = {}
    assert (
        model is not None or tflite_path is not None
    ), "model or tflite_path must be provided"
    if tflite_path is not None:
        metrics = eval_quantized_model_in_tflite(
            test_ds=test_ds, tflite_path=tflite_path
        )

    if model is not None:
        loss, main_metric = model.evaluate(test_ds, verbose=0)
    else:
        loss = -1
        main_metric = -1
    metrics["eval_loss"] = loss
    metrics["eval_metric"] = main_metric
    print(f"{model_name} model metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
        metrics[k] = str(round(v, 3))
    if metrics_file_path is not None:
        if pathlib.Path(metrics_file_path).exists():
            try:
                other_metrics = json.load(open(metrics_file_path, "r"))
            except json.decoder.JSONDecodeError:
                other_metrics = {}
        else:
            other_metrics = {}
        other_metrics[model_name] = metrics
        json.dump(other_metrics, open(metrics_file_path, "w"))
        print(f"Saved metrics to {metrics_file_path}")
    return metrics
