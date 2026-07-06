# Predicting Student Health Risk

**Kaggle Playground Series - Season 6, Episode 7**

## The Challenge
The goal of this competition was to classify students into three health categories (`fit`, `unhealthy`, `at-risk`) based on their lifestyle metrics (sleep, BMI, diet, etc.). The dataset consisted of roughly **690,000 synthetically generated rows**. 

## Data Insights & Challenges
* **Highly Imbalanced Data:** Nearly 86% of the dataset belonged to the `at-risk` class, while `fit` was under 6%. 
* **Evaluation Metric:** Submissions were evaluated on **Balanced Accuracy**. A naive model predicting the majority class would score extremely poorly (~33%).
* **Missing Values:** Almost every column contained 5-12% NaNs. Dropping them was not an option.

## My Approach
1. **Advanced Imputation:** Instead of simple mean/median filling, I utilized sklearn's `IterativeImputer` to predict missing values based on other features. Crucially, I fitted the imputer **only on the training data** and used `.transform()` on the test set to prevent data leakage.
2. **Bayesian Optimization:** Used **Optuna** to tune hyperparameters (depth, learning rate, subsampling) specifically optimizing for Balanced Accuracy.
3. **The "Holy Trinity" Ensemble:** Tabular data responds best to gradient boosting. I trained an Out-Of-Fold (OOF) ensemble using:
   * **XGBoost** (Robustness)
   * **LightGBM** (Leaf-wise speed and depth)
   * **CatBoost** (Symmetric tree strength)
   
All three models predicted probabilities, which were then averaged via **Soft Voting** for the final prediction.

## Results
* **Local OOF Balanced Accuracy:** ~94.24%
* **Kaggle Public Leaderboard:** **0.94386** 

## Lessons Learned
Handling imbalanced classes natively within gradient boosting frameworks (e.g., `class_weight='balanced'`) combined with heterogeneous ensembling provides a massive boost over a single tuned model.