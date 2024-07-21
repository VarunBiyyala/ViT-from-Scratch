
# Vision Transformers from Scratch

This repository contains an implementation of Vision Transformers (ViTs) from scratch. The notebook demonstrates the complete process of setting up, training, and evaluating a Vision Transformer model.

## Table of Contents

- [Installation](#installation)
- [Overview](#overview)
- [Implementation Details](#implementation-details)
  - [Setup](#setup)
  - [Dataset](#dataste)
  - [Image Patching](#image-patching)
  - [Positional Encoding](#positional-encoding)
  - [Transformer Encoder](#transformer-encoder)
  - [Classification Head](#classification-head)
  - [Training Loop](#training-loop)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/VarunBiyyala/ViT-from-Scratch.git

```

## Overview

Vision Transformers (ViTs) are a type of neural network architecture that applies transformer models to image data. Unlike traditional convolutional neural networks (CNNs), ViTs divide images into fixed-size patches and process them as sequences of tokens, enabling the model to capture long-range dependencies in the data.

## Implementation Details

### Setup

The initial step involves setting up the environment and installing necessary libraries. The `einops` library is used for flexible tensor operations and transformations.

```python
!pip install einops
```
### Dataset

For this experimental project, Oxford-IIIT Pet dataset has been downloaded using:
```python
from torchvision.datasets import OxfordIIITPet
```
![Dataset Examples](https://github.com/VarunBiyyala/ViT-from-Scratch/blob/main/ViT_dataste_image.JPG)

### Image Patching

Images are divided into fixed-size patches, which are then flattened and linearly embedded. This step is crucial as it prepares the image data for processing by the transformer model.

### Positional Encoding

Positional encodings are added to the patch embeddings to retain spatial information. This step ensures that the model understands the relative positions of image patches.

### Transformer Encoder

The core component of the Vision Transformer is the Transformer encoder. It consists of multi-head self-attention mechanisms and feed-forward neural networks. This allows the model to process image patches in parallel and capture complex dependencies.

### Classification Head

For classification tasks, a classification head is added to the transformer encoder. This involves taking the output corresponding to the class token (or applying global average pooling) and passing it through a fully connected layer to obtain class logits.

### Training Loop

The training loop includes defining the loss function, optimizer, and running the training process for several epochs. The model's performance is monitored and evaluated using validation data.

### Evaluation

The model's performance is evaluated on a test dataset by computing metrics such as accuracy, which assess how well the model has learned to classify images.

## Results

Though the results documented in this code are not good, i recommend to continue training for more epochs to get better results. My priority is to understand the architecture of ViT's.

## Acknowledgements

Special thanks to the developers of the libraries and tools used in this project, and to the open-source community for their invaluable contributions.
