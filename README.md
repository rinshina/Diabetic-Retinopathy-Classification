# Diabetic-Retinopathy-Classification
## Introduction

Diabetic Retinopathy is a leading cause of blindness in diabetic patients. Early detection through retinal imaging is crucial for preventing severe vision loss. This project focuses on building a convolutional neural network (CNN) to classify retinal images into different stages of Diabetic Retinopathy.

## Dataset

The dataset used for this project consists of retinal images that have been labeled according to the severity of Diabetic Retinopathy. The data is sourced from the [Kaggle Diabetic Retinopathy Detection competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/data).

- **Classes**:
  - 0: No DR
  - 1: Mild
  - 2: Moderate
  - 3: Severe
  - 4: Proliferative DR

## Model Architecture

The model is built using a Convolutional Neural Network (CNN) with the following key layers:
- Convolutional Layers
- Batch Normalization
- Max Pooling
- Fully Connected Layers
- Dropout for regularization

Transfer learning techniques were also applied using pre-trained models like VGG16, ResNet50, and InceptionV3 to improve classification performance.
