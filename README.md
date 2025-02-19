
# Handwritten Equation Recognition using HOG \& SVM

This repository contains a project for recognizing handwritten numerical equations by first segmenting images into meaningful regions (i.e., individual characters) and then classifying them using Histogram of Oriented Gradients (HOG) for feature extraction and a Polynomial Support Vector Machine (SVM) for classification.


---

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Features](#features)
- [Methodology](#methodology)
    - [Data Processing](#data-processing)
    - [HOG Feature Extraction](#hog-feature-extraction)
    - [SVM Classification](#svm-classification)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

---

## Introduction

Handwritten equation recognition poses challenges in dealing with varied handwriting styles and segmentation of dense, grid-like characters. This project tackles these challenges by:

- **Segmenting** full equation images into individual characters.
- **Extracting** HOG features to capture shape and edge details.
- **Classifying** each segmented letter using a polynomial SVM to determine the correct numerical value.

The switch from traditional SIFT to HOG was driven by the need for a fixed-length, efficient descriptor that aligns with the smoothly varying structure of handwritten characters.

---

## Project Overview

The project workflow involves:

- Preprocessing input images (resizing, noise removal).
- Applying adaptive thresholding, contour detection, and morphological operations for robust segmentation.
- Extracting fixed-length feature vectors using HOG.
- Classifying the features with a polynomial SVM.

A modular implementation in PyTorch enables running the model on both CPU and GPU, and it can be easily extended for real-time recognition tasks.

---

## Features

- **Efficient Image Segmentation:** Uses adaptive thresholding and morphological operations to isolate individual characters.
- **HOG Feature Extraction:** Captures essential shape and edge information in a fixed-length feature vector.
- **Polynomial SVM Classifier:** Balances complexity and computational efficiency by leveraging a polynomial kernel.
- **Configurable Parameters:** Easily adjust HOG and SVM parameters to optimize performance.

Below is a comparison of different HOG configurations explored during development:


| Configuration | Orientations | Pixels per Cell | Cells per Block | Performance |
| :-- | :-- | :-- | :-- | :-- |
| **Best Configuration** | 12 | 4 x 4 | 3 x 3 | Optimal performance |
| **Generalized Approach** | 9 | 8 x 8 | 2 x 2 | Underperformed |
| **Overfitting Setup** | 16 | 4 x 4 | 3 x 3 | Overfitting observed |

---

## Methodology

### Data Processing

- **Preprocessing:**
Uniformly resize images and remove noise using morphological operations to enhance segmentation results.
- **Class Balancing:**
Data augmentation techniques ensure that the classifier receives a balanced view of each character category.


### HOG Feature Extraction

- **Gradient Computation:**
Computes gradients for each pixel to capture edge directions.
- **Cell and Block Processing:**
Divides the image into cells, computes local histograms, normalizes over blocks, and flattens into a feature vector.


### SVM Classification

- **Support Vector Machines:**
Utilizes a polynomial kernel SVM to project data into higher dimensions, making it linearly separable.



- **Margin Optimization:**
The classifier maximizes the margin between class boundaries while handling misclassified instances using custom hinge loss.

---



**Clone the Repository:**

```bash
git clone https://github.com/yourusername/SVM_HOG.git

```



## Project Structure



```
.
├── code
│   ├── model.py           # Contains the PolynomialSVM and HingeLoss definitions
│   ├── preprocessing.py   # Functions for image preprocessing and HOG feature extraction
│   └── main.py            # Main testing workflow and integration script
├── saved_features_2
│   └── hog_features.pkl   # Precomputed HOG features and class labels
├── best_svm_model_poly_2.pth  # Trained SVM model weights
└── README.md              # This file
```

---



## License

Distributed under the MIT License. See `LICENSE` for more information.

---

