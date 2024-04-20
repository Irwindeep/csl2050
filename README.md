# Face Identification: Course Project - CSL2050

## Problem Statement

This project aims to perform face identification using various machine-learning techniques. The dataset used is the Labeled Faces in the Wild (LFW) dataset, which contains images of different individuals.

## Prerequisites

To run this project, you'll need to have the following software and libraries installed:

- Python 3. x
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- OpenCV
- Pillow

## Dataset

The LFW dataset is a collection of face images designed to study unconstrained face recognition. The dataset contains more than 13,000 images of 5,749 individuals, with each individual having multiple images captured under varying conditions.

The dataset can be downloaded from the following link: [Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)

## Features

The project explores the following features for face identification:

1. **Histogram of Oriented Gradients (HoG)**: The HoG feature descriptor captures the distribution of intensity gradients within an image, which can be helpful in detecting and describing objects, including faces.

2. **Local Binary Patterns (LBP)**: LBP is a texture descriptor that encodes the local spatial structure of an image by considering the pixel intensities in a neighbourhood around each pixel.

3. **Convolutional Neural Network (CNN) Features**: A pre-trained ResNet-50 model is used to extract deep CNN features from the face images, leveraging deep neural networks' powerful representation learning capabilities.

## Dimensionality Reduction

To reduce the dimensionality of the feature space and improve computational efficiency, the following techniques are employed:

1. **Principal Component Analysis (PCA)**: PCA is used to transform the original feature space into a lower-dimensional subspace while retaining most of the variance in the data.

2. **Linear Discriminant Analysis (LDA)**: LDA is a supervised dimensionality reduction technique that projects the data onto a lower-dimensional subspace while maximising the class separability.

## Machine Learning Models

The project evaluates the performance of various machine learning models for face identification, including:

1. Naive Bayes Classifier
2. K-Nearest Neighbors (KNN)
3. Random Forest Classifier
4. Support Vector Classifier (SVC) with Linear Kernel
5. Artificial Neural Network (ANN)

The dataset is split into training and testing sets, and the models are trained on the training data and evaluated on the testing data.

## Usage

1. Clone the repository or download the project files.
2. Install the required dependencies (e.g., NumPy, Pandas, Scikit-learn, PyTorch, OpenCV, Pillow).
3. Download the LFW dataset and place it in the appropriate directory.
4. Run the provided Jupyter Notebook or Python script to preprocess the data, extract features, perform dimensionality reduction, train the models, and evaluate their performance.

## Team Members

1. Irwindeep Singh (B22AI022)
2. Ramninder Singh (B22AI032)
3. Sarthak Malviya (B22AI034)
4. Vikrant Singh (B22AI043)
5. Paras Govind Kalyanpad (B22CS037)