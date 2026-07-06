# MNIST Digit Recognizer
## The Challenge
The classic Computer Vision problem: identifying handwritten digits (0-9) from 28x28 pixel grayscale images. Kaggle provides the data in a flattened CSV format (784 pixels per row).

## My Approach (Two-fold Experiment)

### 1. Traditional Computer Vision (HOG + SVM)
Before jumping into Deep Learning, I wanted to understand how feature extraction works. 
* I reshaped the flattened 784-pixel arrays back into 28x28 images.
* Extracted **HOG (Histogram of Oriented Gradients)** features. This expanded the dataset to 1296 directional features, transforming raw pixels into "smart" edge-detection vectors.
* Trained an **SVM** on these features, achieving excellent accuracy extremely fast.

### 2. Deep Learning (PyTorch Lightning CNN)
To maximize accuracy, I built a Convolutional Neural Network (CNN) from scratch using **PyTorch Lightning**.
* **Architecture:** Two Convolutional blocks (Conv2d -> ReLU -> MaxPool2d) followed by a flattened Linear classifier with Dropout.
* **Why CNN?** Unlike linear models, the CNN maintains spatial awareness and achieves translation invariance through Max Pooling (recognizing a "1" even if it's shifted a few pixels to the side).
* **Hardware:** Utilized Kaggle's T4x2 GPU accelerators to drastically reduce training time.

## Lessons Learned
This project was a great bridge from tabular data to computer vision. It highlighted why flattening images destroys spatial relationships, and why convolutional kernels (sliding windows) are the superior mathematical approach for image data.