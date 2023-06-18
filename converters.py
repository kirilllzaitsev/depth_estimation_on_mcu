import os

import numpy as np
import tensorflow as tf


class Converter:
    def __init__(self, cfg):
        self.cfg = cfg

    def keras_to_tflite(self, fp_model, model_name, do_return_path=False):
        # Convert the model to TFLite without quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(fp_model)
        fp_tflite_model = converter.convert()

        # Save the model to disk
        save_path = f"{self.cfg.save_model_dir}/{model_name}.tflite"
        open(save_path, "wb").write(fp_tflite_model)
        print(f"Model was saved at location: {os.path.abspath(save_path)}")

        # Show the model size for the non-quantized HDF5 model
        fp_h5_in_kb = (
            os.path.getsize(f"{self.cfg.save_model_dir}/{model_name}.h5") / 1024
        )
        print("HDF5 Model size: %d KB" % fp_h5_in_kb)

        # Show the model size for the non-quantized TFLite model
        fp_tflite_in_kb = (
            os.path.getsize(f"{self.cfg.save_model_dir}/{model_name}.tflite") / 1024
        )
        print("TFLite Model size: %d KB" % fp_tflite_in_kb)

        # Determine the reduction in model size
        print(
            "\nReduction in file size by a factor of %f"
            % (fp_h5_in_kb / fp_tflite_in_kb)
        )
        if do_return_path:
            return fp_tflite_model, f"{self.cfg.save_model_dir}/{model_name}.tflite"
        return fp_tflite_model

    def dynamic_range_quantization(self, fp_model, model_name):
        # Convert the model to TFLite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(fp_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        dynR_quant_tflite_model = converter.convert()

        # Save the model to disk
        open(f"{self.cfg.save_model_dir}/{model_name}.tflite", "wb").write(
            dynR_quant_tflite_model
        )

        print(
            "Model was saved at location: %s"
            % os.path.abspath(
                f"{self.cfg.save_model_dir}/{model_name}.tflite"
            )
        )
        return dynR_quant_tflite_model

    def check_quantized_model(self, fp_model):
        interpreter = tf.lite.Interpreter(model_content=fp_model)
        input_type = interpreter.get_input_details()[0]["dtype"]
        print("input: ", input_type)
        output_type = interpreter.get_output_details()[0]["dtype"]
        print("output: ", output_type)

    def eight_bit_quantization(self, fp_model, train_ds, model_name):
        def representative_data_gen():
            for input_value, depth in train_ds.take(100):
                if hasattr(input_value, "shape"):
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
        open(f"{self.cfg.save_model_dir}/{model_name}.tflite", "wb").write(
            tflite_model_quant_int8
        )

        print(
            "Model was saved at location: %s"
            % os.path.abspath(f"{self.cfg.save_model_dir}/{model_name}.tflite")
        )

        # Show the model size for the 8-bit quantized TFLite model
        tflite_quant_in_kb = (
            os.path.getsize(f"{self.cfg.save_model_dir}/{model_name}.tflite") / 1024
        )
        print("TFLite Model size with 8-bit quantization: %d KB" % tflite_quant_in_kb)
        fp_tflite_in_kb = (
            os.path.getsize(f"{self.cfg.save_model_dir}/depth_model_fp32.tflite") / 1024
        )

        print("TFLite Model size without quantization: %d KB" % fp_tflite_in_kb)

        # Determine the reduction in model size
        print(
            "\nReduction in model size by a factor of %f"
            % (fp_tflite_in_kb / tflite_quant_in_kb)
        )

        return tflite_model_quant_int8
