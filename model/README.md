# CycleGAN for Image-to-Image Translation (Style Transfer)
This Model was taken from the CycleGan repository (see Acknowledgments), and all files were combined into one. The files are still distinguished by comments seperating them. They can be split up again if needed! The README will include information that is not necessarry in the one-file-state.

## Overview
This folder contains an implementation of CycleGAN, a deep learning model for unpaired image-to-image translation. The code is built using PyTorch and is designed to translate images from one domain to another without the need for paired training data.

## Features
- __CycleGAN Model__: Implementation of the CycleGAN architecture with two generators and two discriminators for image translation.
- __Loss Functions__: Adversarial loss, cycle consistency loss, and identity loss are used to train the generators.
- __Training and Testing__: Training loop for model training and a testing function for evaluating the trained models on a separate dataset.
- __Learning Rate Schedulers__: Learning rate schedulers for both the generator and discriminator networks.

## Requirements
- PyTorch
- torchvision
- numpy
- Pillow

Other dependencies (specified in requirements.txt)

## Usage

### Install Dependencies:

```bash
pip install numpy Pillow torch torchvision
```

### Running:
In one-file-mode this will do traning and testing

```bash
python cygan.py
```


### Configuration

- Hyperparameters: Hyperparameters such as batch size, learning rate, and the number of residual blocks can be configured in cycle_gan.py.
- Dataset: The dataset path and image transformations can be adjusted in cycle_gan.py.

## Results
- Training results, including generated images and model checkpoints, will be saved in the images/ and save/ directories.

## Acknowledgments
This implementation is based on the original CycleGAN paper: https://arxiv.org/abs/1703.10593
    
Using [this branch](https://github.com/Lornatang/CycleGAN-PyTorch/tree/88e1f17a8a7be24b982cce16a6cf0db043e13bdc) to be more specific.