import os

import tensorflow as tf
from config import cfg


def keras_to_tflite(fp_model):
    # Convert the model to TFLite without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(fp_model)
    fp_tflite_model = converter.convert()

    # Save the model to disk
    open(f"{cfg.save_model_dir}/depth_model_f32.tflite", "wb").write(fp_tflite_model)

    # Show the model size for the non-quantized HDF5 model
    fp_h5_in_kb = os.path.getsize(f"{cfg.save_model_dir}/depth_model_f32.h5") / 1024
    print("HDF5 Model size without quantization: %d KB" % fp_h5_in_kb)

    # Show the model size for the non-quantized TFLite model
    fp_tflite_in_kb = (
        os.path.getsize(f"{cfg.save_model_dir}/depth_model_f32.tflite") / 1024
    )
    print("TFLite Model size without quantization: %d KB" % fp_tflite_in_kb)

    # Determine the reduction in model size
    print(
        "\nReduction in file size by a factor of %f" % (fp_h5_in_kb / fp_tflite_in_kb)
    )
    return fp_tflite_model


def dynamic_range_quantization(fp_model):
    # Convert the model to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(fp_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dynR_quant_tflite_model = converter.convert()

    # Save the model to disk
    open(f"{cfg.save_model_dir}/depth_model_quant8_dynR.tflite", "wb").write(
        dynR_quant_tflite_model
    )

    print(
        "Model was saved at location: %s"
        % os.path.abspath(f"{cfg.save_model_dir}/depth_model_quant8_dynR.tflite")
    )
    return dynR_quant_tflite_model


def check_quantized_model(fp_model):
    interpreter = tf.lite.Interpreter(model_content=fp_model)
    input_type = interpreter.get_input_details()[0]["dtype"]
    print("input: ", input_type)
    output_type = interpreter.get_output_details()[0]["dtype"]
    print("output: ", output_type)


def eight_bit_quantization(fp_model, train_images, model_name):
    def representative_data_gen():
        for input_value in (
            tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100)
        ):
            if len(input_value.shape) == 3:
                input_value = input_value[..., np.newaxis]
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(fp_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant_int8 = converter.convert()

    # Save the quantized model to disk
    open(f"{cfg.save_model_dir}/{model_name}.tflite", "wb").write(
        tflite_model_quant_int8
    )

    print(
        "Model was saved at location: %s"
        % os.path.abspath(f"{cfg.save_model_dir}/{model_name}.tflite")
    )

    # Show the model size for the 8-bit quantized TFLite model
    tflite_quant_in_kb = (
        os.path.getsize(f"{cfg.save_model_dir}/{model_name}.tflite") / 1024
    )
    print("TFLite Model size with 8-bit quantization: %d KB" % tflite_quant_in_kb)
    fp_tflite_in_kb = (
        os.path.getsize(f"{cfg.save_model_dir}/depth_model_f32.tflite") / 1024
    )

    print("TFLite Model size without quantization: %d KB" % fp_tflite_in_kb)

    # Determine the reduction in model size
    print(
        "\nReduction in model size by a factor of %f"
        % (fp_tflite_in_kb / tflite_quant_in_kb)
    )

    return tflite_model_quant_int8
