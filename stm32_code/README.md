# Depth estimation on MCU

## STM32

### Setup

Please install packages from requirements.txt

### Run

full_pipeline.ipynb is a notebook that trains all model types and stores them under models/, cfiles/ folders.

## MAX78000

ai8x-training and ai8x-synthesis are clones in full to facilitate reproducibility of experiments. Personal contributions are as follows:

- ai8x-synthesis
  - networks/ai85depthunetmedium.yaml
  - synthed_net/* - synthesized networks (one from an official demo) for depth estimation (\*depth\*) and camera communication (camvid_unet)
    - to test it, flash aisegment_unet-demo on the device and run camvid_unet/Utility/SerialLoader.py
    - ensure the paths to inputs exist
  - notebooks/*[^Parse_sampleout_and_compare_SVHN.ipynb] - off-device validation of networks and some tooling
- trained/ai85-cifar10-qat8.pth.tar was used for comparison with NAS results
- ai8x-training
  - *_artifacts - train & eval results for the networks with corresponding names
  - ai8x_datasets/ade20k.py - ADE20K dataset for initial (before depth estimation) testing with semantic segmentation
  - ai8x_datasets/nyuv2.py - NYUv2 dataset
  - logs/2023.06.18-080333 - logs from completed NAS
  - models/depth-est-unet.py - depth estimation UNets
  - models/test-depth-est-mobilenet-v2.py - initial attempt to bring MobileNetV2 into the depth estimation pipeline
  - policies/qat_policy_ai85depthunet.yaml - scheduling policy for training depth UNets
  - NAS
    - nas/nas_policy_cifar10_reduced.yaml - NAS definition file
    - models/ai85nasnet-sequential.py - add CIFAR10 network
