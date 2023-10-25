# Landscape Image Prediction 

## Overview
The Landscape Image Prediction code is designed to recognize and classify different types of landscape images, be it mountains, beaches, forests, etc. Leveraging the power of a pre-trained MobileNetV2 model, this script fine-tunes the model to the specific task of landscape classification using a dataset fetched from Zenodo.

## Dataset
The data consists of landscape images categorized into different classes. It is divided into a training set and a validation set, which can be downloaded from the given Zenodo URLs in the script. Each class in the dataset represents a unique landscape type.

## Model Architecture
The script uses the MobileNetV2 model pre-trained on the ImageNet dataset. The base model is extended with a series of preprocessing layers for data augmentation and a dense output layer for classification. The architecture ensures efficient training while maintaining high accuracy.

* Pre-processing Layers: These layers include rescaling to normalize pixel values, random flipping, and random rotation to augment the input images, enriching the dataset without the need for more data.
* Base Model: The MobileNetV2 model, which is pre-trained on the ImageNet dataset. Its training is turned off to utilize the previously learned features.
* Pooling and Flattening: MaxPooling is employed for spatial data reduction followed by a Flatten layer to transform the data into a format suitable for the dense layer.
* Output Layer: A dense layer designed with units equal to the number of landscape classes. The softmax activation function is used to output the class probabilities.
  
## Performance
The model has demonstrated robust performance over the course of training. Specifically, by the end of 20 epochs, the model achieved the following:

**Training Data:**
- Loss: 0.8459
- Accuracy: 86.05%

**Validation Data:**
- Loss: 0.9171
- Accuracy: 85.93%
  
The consistently high accuracy on both the training and validation sets indicates the model's capacity to generalize well on unseen data, making it reliable for landscape image predictions.
