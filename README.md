# Depth estimation on MCU

This document describes STM32, MAX78000, and NAS parts of the project.

## STM32

### Setup

```pip install -r requirements.txt```

Replace absolute paths and set a proper serial port for communication.

### Run

full_pipeline.ipynb is a notebook that trains all model types and stores them under models/ and cfiles/ folders.

models/ contains trained models, Tensorflow history files, and loss curves (can also be generated via plot_utils.py using a history file).

### Results

Trained models are stored in the models/ folder. models_64x64 contains results for 64x64x3 inputs.

depth_inference.py is used for interfacing with a device and prediction.png is a sample prediction. A highly structured pattern observed in prediction.png does not indicate an issue with the model, but rather a flaw with data I/O.

## MAX78000

### Setup

```pip install -r requirements.txt```

```bash
sudo apt install -y openocd libxcb-glx0 libxcb-icccm4 libxcb-image0 libxcb-shm0 libxcb-util1 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-render0 libxcb-shape0 libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 libxcb1 libxkbcommon-x11-0 libxkbcommon0 libgl1 libusb-0.1-4 libhidapi-libusb0 libhidapi-hidraw0
```

Replace absolute paths and set a proper serial port for communication.

### Contributions

ai8x-training and ai8x-synthesis are clones in full to facilitate reproducibility of experiments. Personal contributions are the following:

- ai8x-synthesis
  - networks/ai85depthunetmedium.yaml
  - synthed_net/* - synthesized networks (one from an official demo) for depth estimation (\*depth\*) and camera communication (camvid_unet)
    - to test it, flash aisegment_unet-demo on the device and run camvid_unet/Utility/SerialLoader.py
    - ensure the paths to inputs exist
  - notebooks/*[^Parse_sampleout_and_compare_SVHN.ipynb] - off-device validation of networks and some tooling
- ai8x-training
  - *_artifacts - train & eval results for the networks with corresponding names
  - ai8x_datasets/ade20k.py - ADE20K dataset for initial (before depth estimation) testing with semantic segmentation
  - ai8x_datasets/nyuv2.py - NYUv2 dataset
  - models/depth-est-unet.py - depth estimation UNets
  - models/test-depth-est-mobilenet-v2.py - initial attempt to bring MobileNetV2 into the depth estimation pipeline
  - policies/qat_policy_ai85depthunet.yaml - scheduling policy for training depth UNets
  - NAS
    - ai8x-training/logs/2023.06.18-080333 - logs from a completed two-stage NAS pipeline
    - nas/nas_policy_cifar10_reduced.yaml - NAS definition file
    - models/ai85nasnet-sequential.py - add CIFAR10 network
    - trained/ai85-cifar10-qat8.pth.tar was used for comparison with NAS results

### Results

To get the results, uncomment a code block in 'makefile' that corresponds to the model of interest and run 'make run-pipeline' in the root of ai8x-synthesis. The result of the run will be in the 'synthed_net' folder.

Latency of the medium-sized UNet is obtained by flashing the synthed_net/ai85depthunetmedium application on the device and monitoring a serial port. Large-sized UNet does not fit on the device.

Visual interfacing with the board is explored for image segmentation case under the synthed_net/camvid_unet folder, but is not integrated into the depth estimation pipeline.
