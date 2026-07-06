# Titanic: Machine Learning from Disaster

## The Challenge
The legendary Titanic dataset. The goal is to predict passenger survival based on features like age, sex, ticket class, and fare.

## Feature Engineering
Instead of just feeding raw data into models, I focused heavily on Feature Engineering:
* **Title Extraction:** Extracted titles (Mr, Mrs, Miss, Master, Rare) from passenger names to better estimate age and social status.
* **Group Size:** Calculated family/group size from `Ticket` frequencies.
* **Fare Per Person:** Divided the raw `Fare` by `Group_Size` to reveal the *true* socio-economic status of passengers traveling on group tickets.
* **Cabin Logic:** Treated missing cabins not as "errors", but as meaningful information (passengers without a recorded cabin had a significantly lower survival rate).

## Model Comparison
I experimented with multiple architectures:
1. K-Nearest Neighbors (KNN)
2. Support Vector Machines (SVM with RBF kernel)
3. Shallow Neural Network (PyTorch)
4. XGBoost (Tuned via Optuna)

## Lessons Learned
While my XGBoost model dominated on the 690k-row Student Health dataset, it struggled with **overfitting** on the tiny 891-row Titanic dataset (scoring ~74.8%). In contrast, models that create smoother decision boundaries, like **SVM** and **Neural Networks**, generalized much better on this small dataset, achieving over **76% accuracy** on the public leaderboard.