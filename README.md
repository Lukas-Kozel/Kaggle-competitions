# Kaggle Machine Learning Portfolio

This repository contains my solutions and approaches to various Kaggle competitions. My goal is to showcase a progression from classic tabular data analysis to advanced ensemble methods and deep learning.

## Repository Structure

| Project | Domain | Key Technologies & Concepts |
| :--- | :--- | :--- |
| [**1. Titanic: Machine Learning from Disaster**](./titanic) | Tabular / Binary Classification | EDA, Feature Engineering, SVM, KNN |
| [**2. Predicting Student Health Risk**](./student-health-risk) | Tabular / Multiclass / Imbalanced | `IterativeImputer`, Optuna, XGBoost, LightGBM, CatBoost, Ensembling |
| [**3. MNIST Digit Recognizer**](./mnist-dataset) | Computer Vision / Deep Learning | HOG Feature Extraction, PyTorch Lightning, CNN Architecture |

## Core Skills Demonstrated
* **Data Preprocessing & Feature Engineering:** Handling missing data without leakage (`IterativeImputer`), categorical encoding, handling synthetic artifacts.
* **Hyperparameter Tuning:** Bayesian optimization using **Optuna**.
* **Ensemble Learning:** OOF (Out-Of-Fold) validation, Soft-voting blending (The "Holy Trinity": XGBoost + LightGBM + CatBoost).
* **Deep Learning:** Building and training Convolutional Neural Networks (CNNs) using **PyTorch Lightning** on GPUs.

*Feel free to explore the subfolders for detailed breakdowns of each project!*