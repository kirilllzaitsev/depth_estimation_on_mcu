# Depth estimation on MCU

This repository contains Tensorflow code to create, train, and deploy a monocular depth estimation model for an STM32 microcontroller.

## Setup

- ```pip install -r requirements.txt```
- create a .env file in the project root with the following variables:

```bash
path_to_project_dir=/abs/path/to/project/root
base_dataset_dir=/abs/path/to/nyuv2/dataset
```

Please, replace absolute paths and set a proper serial port for communication in depth_inference.py

## Run

full_pipeline.ipynb is a notebook that trains models with multiple quantization and pruning techniques, storing them under models/ and cfiles/ folders relative to the project root.

## Results

Trained models will appear in the models/ folder. .h5 or .tflite files can be used within STM32 to profile the model and validate its performance in an environment similar to the real MCU.

cfiles/ contains C files for each model type. Flashing a model to a device requires only its corresponding header file in cfiles/.

depth_inference.py is used for interfacing with a device and prediction.png is a sample prediction. A highly structured pattern observed in prediction.png does not indicate an issue with the model, but rather a flaw with data I/O.
