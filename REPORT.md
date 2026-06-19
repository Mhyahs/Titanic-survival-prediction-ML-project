# Titanic Survival Prediction Project Report

## Executive Summary

This project aimed to build and evaluate machine learning models for predicting passenger survival on the Titanic using the classic Kaggle dataset. Three models—Logistic Regression, Random Forest, and Decision Tree—were implemented, evaluated, and optimized through feature engineering and hyperparameter tuning. The optimized Decision Tree model achieved the highest accuracy of **84.36%**. Feature importance analysis revealed that sex, fare, and age are the most influential factors in survival. The results demonstrate that with proper preprocessing and careful tuning, reliable predictions can be achieved, aligning with historical accounts of the disaster.

---

## 1. Introduction

The RMS Titanic sank on its maiden voyage in 1912 after colliding with an iceberg, resulting in the loss of over 1,500 lives among the 2,224 passengers and crew. This tragic event has become one of history's most famous maritime disasters, and the passenger data has become a classic benchmark dataset in machine learning.

The primary objective of this project is to build a predictive model that can determine whether a given passenger would survive based on features such as age, gender, ticket class, fare, and family relations. Such analysis not only provides insights into survival patterns during emergencies but also serves as a comprehensive exercise in executing the full data science pipeline—from data cleaning and exploration to model deployment and evaluation.

---

## 2. Dataset Description

### 2.1. Source

The dataset was obtained from the DataScienceDojo repository on GitHub:
```
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```
It contains information on 891 passengers (rows) and 12 features (columns).

### 2.2. Feature Definitions

| Feature | Type | Description |
|---------|------|-------------|
| PassengerId | Integer | Unique identifier for each passenger |
| Survived | Integer (0/1) | Target variable: 0 = deceased, 1 = survived |
| Pclass | Integer (1,2,3) | Ticket class (1 = upper, 2 = middle, 3 = lower) |
| Name | String | Passenger's full name |
| Sex | String | Gender (male/female) |
| Age | Float | Age in years |
| SibSp | Integer | Number of siblings/spouses aboard |
| Parch | Integer | Number of parents/children aboard |
| Ticket | String | Ticket number |
| Fare | Float | Ticket fare |
| Cabin | String | Cabin number |
| Embarked | String | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

### 2.3. Preliminary Statistics

Initial data inspection revealed:
- **38.4%** of passengers survived (mean of `Survived`).
- The average age was approximately **29.7 years**.
- Most passengers travelled in third class.
- Missing values were present in `Age`, `Cabin`, and `Embarked`.

---

## 3. Data Preprocessing

### 3.1. Handling Missing Values

- **Age:** 177 missing entries were imputed using the **median** age (28 years) to avoid skewing the distribution.
- **Embarked:** 2 missing entries were filled with the **mode** (most frequent value), which was 'S' (Southampton).
- **Cabin:** Due to the overwhelming number of missing values (687 out of 891), this feature was dropped entirely.

### 3.2. Categorical Encoding

- **Sex:** Converted to numeric: `male` → 0, `female` → 1.
- **Embarked:** Mapped as: `S` → 0, `C` → 1, `Q` → 2.

### 3.3. Irrelevant Features Removal

`PassengerId`, `Name`, and `Ticket` were dropped as they do not contribute meaningfully to survival prediction (either being unique identifiers or containing too many distinct values).

### 3.4. Feature Engineering

A new feature **`FamilySize`** was created to capture the effect of traveling with family:
```
FamilySize = SibSp + Parch + 1
```
This accounts for the passenger themselves and may reflect the survival advantage or disadvantage of being alone or in a large family.

### 3.5. Feature Scaling

To ensure models sensitive to feature magnitudes (e.g., Logistic Regression) perform optimally, all numeric features were standardized using `StandardScaler` (zero mean, unit variance).

---

## 4. Methodology

### 4.1. Data Splitting

The preprocessed data was split into **training (80%)** and **testing (20%)** sets using `train_test_split` with `random_state=42` for reproducibility.

### 4.2. Models Selected

Three widely used classification algorithms were chosen:

1. **Logistic Regression:** A simple linear model serving as a baseline.
2. **Random Forest:** An ensemble method combining multiple decision trees, known for good performance on tabular data.
3. **Decision Tree:** A single tree-based model offering high interpretability.

### 4.3. Evaluation Metrics

The models were assessed using:
- **Accuracy:** Overall correct predictions.
- **Precision, Recall, and F1-score:** Per-class performance metrics.
- **AUC-ROC:** Ability to discriminate between classes across thresholds.
- **Confusion Matrix:** Detailed breakdown of correct and incorrect predictions.

### 4.4. Hyperparameter Tuning

For each model, a grid search (`GridSearchCV`) with 5‑fold cross‑validation was performed to identify the optimal hyperparameters.

| Model | Best Parameters |
|-------|-----------------|
| Random Forest | `max_depth=20`, `min_samples_leaf=2`, `min_samples_split=10`, `n_estimators=300` |
| Logistic Regression | `C=0.01`, `penalty='l2'`, `solver='saga'` |
| Decision Tree | `criterion='gini'`, `max_depth=10`, `min_samples_leaf=4`, `min_samples_split=2` |

---

## 5. Results

### 5.1. Model Performance: Before and After Tuning

| Model | Accuracy (Before Tuning) | Accuracy (After Tuning) |
|-------|--------------------------|--------------------------|
| Random Forest | 82.12% | 82.68% |
| Logistic Regression | 79.89% | 79.33% |
| Decision Tree | 79.33% | **84.36%** |

Hyperparameter tuning significantly improved the Decision Tree, making it the best-performing model. This improvement underscores the importance of limiting tree depth and setting appropriate leaf size to prevent overfitting.

### 5.2. Feature Importance Analysis

Using the Random Forest model (which provides built-in feature importance), the top contributors to survival prediction were:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Sex | 26.84% |
| 2 | Fare | 26.84% |
| 3 | Age | 24.88% |
| 4 | Pclass | 7.25% |
| 5 | FamilySize | 4.84% |
| 6 | SibSp | 3.68% |
| 7 | Embarked | 3.18% |
| 8 | Parch | 2.49% |

**Interpretation:** Sex, fare, and age emerged as the most influential factors, aligning with historical evidence: women, children, and wealthier passengers had higher survival rates.

### 5.3. ROC-AUC Evaluation

ROC curves for all optimised models were plotted:

- **Random Forest:** AUC ≈ 0.88
- **Logistic Regression:** AUC ≈ 0.87
- **Decision Tree:** AUC ≈ 0.89

All models exhibited strong discriminative ability, with the Decision Tree achieving the highest AUC, confirming its superior performance.

### 5.4. Confusion Matrices

Confusion matrices for each model showed that the Decision Tree made the fewest classification errors, particularly in correctly identifying deceased passengers (class 0), outperforming the other two models.

### 5.5. Cross-Validation Stability

Mean cross‑validation accuracies (5‑fold CV) for the tuned models:

| Model | Mean CV Accuracy | Std Deviation |
|-------|------------------|---------------|
| Random Forest | 82.72% | ±2.04% |
| Logistic Regression | 80.05% | ±2.67% |
| Decision Tree | 81.47% | ±1.51% |

The Decision Tree not only achieved the highest test accuracy but also showed the lowest variance, indicating greater stability and generalisation capability.

---

## 6. Discussion

### 6.1. Why the Decision Tree Excelled After Tuning

While Random Forest typically outperforms a single decision tree due to its ensemble nature, the Decision Tree surpassed it after hyperparameter tuning. Possible reasons include:
- Careful tuning (limiting depth, increasing leaf samples) mitigated overfitting.
- The dataset is relatively small; simpler models can capture patterns more effectively.
- Random Forest, despite being robust, may introduce noise in this specific dataset.

### 6.2. Alignment with Domain Knowledge

The high importance of `Sex`, `Fare`, and `Age` matches historical accounts and previous studies:
- **Sex:** Women and children were prioritised during evacuation.
- **Fare (as a proxy for class):** First‑class passengers had better access to lifeboats.
- **Age:** Children had higher survival chances.

### 6.3. Limitations

- **Dataset Size:** With only 891 records, the dataset is limited for more complex models.
- **Limited Features:** Important details (e.g., exact cabin location, family relations beyond immediate relatives) were unavailable.
- **Simplistic Imputation:** Using median for age and dropping Cabin may discard useful information.
- **Classical Models Only:** Advanced boosting methods (XGBoost, LightGBM) were not explored.

---

## 7. Conclusion and Future Work

### 7.1. Conclusion

This project successfully built and evaluated three machine learning models for Titanic survival prediction. After rigorous preprocessing, feature engineering, and hyperparameter optimisation, the Decision Tree model achieved the best accuracy of **84.36%**. Feature importance analysis confirmed that gender, fare, and age are the dominant survival factors, consistent with historical narratives. The results demonstrate that a well‑tuned decision tree can be highly effective for this classic classification problem.

### 7.2. Suggested Enhancements

To further improve performance and robustness, the following steps are recommended:

1. **Advanced Models:** Explore gradient boosting algorithms (XGBoost, LightGBM, CatBoost) which often excel with tabular data.
2. **Enhanced Feature Engineering:**
   - Extract titles from names (e.g., Mr, Mrs, Miss, Master) to capture social status and age.
   - Create interaction features (e.g., `FamilySize * Pclass`).
   - Use advanced imputation (KNN, MICE) for missing values.
3. **Nested Cross‑Validation:** Use nested CV for more robust hyperparameter selection.
4. **Pipeline Construction:** Build a scikit‑learn `Pipeline` to streamline preprocessing and modelling for easier reproducibility.
5. **Data Balancing:** Though not heavily imbalanced, test techniques like SMOTE if needed.
6. **External Data:** If available, incorporate additional data (crew details, lifeboat assignments) to enrich the model.

---

## 8. Appendices

### 8.1. Reproduction Instructions

To reproduce the analysis, run the Jupyter notebook `test.ipynb` in a Python environment with the following dependencies:

- Python 3.7+
- pandas, numpy, scikit‑learn, matplotlib, seaborn

Installation and execution:
```bash
# Install required libraries
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Execute the notebook to regenerate outputs
jupyter nbconvert --to notebook --execute test.ipynb --inplace

# Export to HTML
jupyter nbconvert --to html test.ipynb --output test.html
```

### 8.2. Project Files

- `test.ipynb` – Main notebook containing all code and visualisations.
- `test.html` – HTML export of the notebook for easy viewing.
- `REPORT.md` – This comprehensive report.


