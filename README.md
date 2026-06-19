# Titanic Survival Prediction — Machine Learning Classification Project

A comprehensive machine learning project that predicts passenger survival on the Titanic using multiple classification models and rigorous evaluation methodology. This project demonstrates the complete data science workflow: data exploration, preprocessing, feature engineering, model training, hyperparameter optimization, and detailed performance analysis.

## Executive Summary

This project builds and evaluates three machine learning models (Logistic Regression, Random Forest, and Decision Tree) to predict passenger survival on the Titanic. After careful preprocessing, feature engineering, and hyperparameter tuning, the **Decision Tree achieved the highest accuracy of 84.36%**. Feature importance analysis revealed that **sex, fare, and age** are the most influential factors in survival prediction. For detailed findings and methodology, see the [REPORT.md](REPORT.md).

## Overview

The RMS Titanic sank in 1912, resulting in the loss of over 1,500 lives. This tragedy has become one of history's most famous maritime disasters, and the passenger dataset a classic benchmark in machine learning. This project analyzes survival patterns using classification models to determine whether a given passenger would survive based on features such as:
- **Demographic factors**: Age, Sex (Gender)
- **Travel class**: Ticket class (1st, 2nd, 3rd)
- **Family relations**: Number of siblings/spouses, parents/children aboard
- **Economic status**: Ticket fare
- **Embarkation**: Port of boarding

## Dataset

- **Source**: [DataScienceDojo Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Samples**: 891 passengers with 12 features
- **Target Variable**: Survived (binary: 0 = deceased, 1 = survived)
- **Survival Rate**: 38.4% of passengers survived
- **Key Features**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, and engineered FamilySize

## Key Results

### Model Performance After Hyperparameter Tuning

| Model | Accuracy | AUC-ROC | Cross-Val Std Dev |
|-------|----------|---------|-------------------|
| **Decision Tree** | **84.36%** | **0.89** | **±1.51%** |
| Random Forest | 82.68% | 0.88 | ±2.04% |
| Logistic Regression | 79.33% | 0.87 | ±2.67% |

The Decision Tree model achieved the best performance with the lowest variance, indicating superior generalization.

### Feature Importance (Top 5)

1. **Sex** (26.84%) — Gender was the strongest survival predictor
2. **Fare** (26.84%) — Ticket price/class significance  
3. **Age** (24.88%) — Age demographics impact
4. **Pclass** (7.25%) — Ticket class influence
5. **FamilySize** (4.84%) — Traveling with family effects

## Project Structure

```
titanic/
├── test.ipynb                    # Main Jupyter notebook with full analysis and code
├── test.html                     # HTML export of the notebook for easy viewing
├── REPORT.md                     # Comprehensive project report (methodology & results)
├── README.md                     # This file
├── requirements.txt              # Python package dependencies
└── test_files/                   # Supporting files (if any)
```

- **test.ipynb**: Complete implementation of the ML pipeline with visualizations
- **REPORT.md**: Detailed technical report with discussion and future recommendations
- **README.md**: Quick-start guide and project overview (this file)

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip or conda package manager
- Jupyter Notebook (for interactive exploration)

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd titanic
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `pandas` – Data manipulation and analysis
- `numpy` – Numerical computing
- `scikit-learn` – Machine learning models and preprocessing
- `matplotlib` – Data visualization
- `seaborn` – Statistical visualizations
- `jupyter` – Interactive notebook environment

## Usage

### Run the Interactive Notebook
```bash
jupyter notebook test.ipynb
```
This opens the notebook in your browser for interactive exploration and execution.

### Execute and Export the Notebook
```bash
# Run all cells and regenerate outputs
jupyter nbconvert --to notebook --execute test.ipynb --inplace

# Export to HTML for viewing in a browser
jupyter nbconvert --to html test.ipynb --output test.html
```

### Python Script Execution
```bash
# Export notebook as Python script
jupyter nbconvert --to script test.ipynb --output analysis.py

# Run the script
python analysis.py
```

## Methodology

### Data Preprocessing
- **Missing Value Handling**:
  - Age: Imputed with median (28 years) from 177 missing values
  - Embarked: Filled with mode ('S' Southampton) from 2 missing values
  - Cabin: Dropped entirely (687 out of 891 missing)
- **Categorical Encoding**:
  - Sex: male=0, female=1
  - Embarked: S=0, C=1, Q=2
- **Feature Removal**: PassengerId, Name, Ticket (non-predictive)
- **Feature Scaling**: StandardScaler (zero mean, unit variance)

### Feature Engineering
- Created **FamilySize** = SibSp + Parch + 1
- Captures effects of traveling with family on survival

### Model Selection & Training
**Three classification algorithms**:
1. **Logistic Regression** – Linear baseline model
2. **Random Forest** – Ensemble tree-based model (100+ trees)
3. **Decision Tree** – Interpretable single-tree model

**Hyperparameter Optimization**:
- GridSearchCV with 5-fold cross-validation
- Best parameters identified for each model
- Evaluated on 20% holdout test set

### Evaluation Metrics
- **Accuracy** – Overall prediction correctness
- **Precision & Recall** – Per-class performance
- **F1-Score** – Harmonic mean of precision/recall
- **ROC-AUC** – Discriminative ability across thresholds
- **Confusion Matrix** – Detailed error breakdown

## Key Visualizations

The notebook includes the following visualizations:
- **Feature Importance Plot** – Bar chart of Random Forest feature importance
- **Confusion Matrices** – Classification errors for each model
- **ROC-AUC Curves** – Model discrimination performance
- **Cross-Validation Distributions** – Model stability boxplots
- **Model Accuracy Comparison** – Before/after tuning comparison chart
- **Survival Distribution** – Target variable statistics

## Results Interpretation

### Why Decision Tree Excelled
Despite Random Forest typically outperforming single decision trees, the Decision Tree achieved the highest accuracy after tuning due to:
- Careful hyperparameter selection (limiting depth, increasing leaf samples) mitigated overfitting
- Small dataset size (891 samples) favors simpler, interpretable models
- Ensemble methods can introduce noise in limited-sample scenarios

### Domain Alignment
Feature importance findings align with historical evidence:
- **Sex**: Women and children were prioritized during evacuation ("women and children first")
- **Fare**: First-class passengers had better access to lifeboats and crew assistance
- **Age**: Children had higher survival chances due to evacuation priority

## Limitations

- **Dataset Size**: Only 891 records; limited for very complex models
- **Feature Scope**: Missing potentially important details (cabin location, lifeboat assignments, crew data)
- **Imputation Strategy**: Median imputation for Age and dropping Cabin may discard useful information
- **Model Diversity**: Limited to classical ML models; advanced boosting methods not explored
- **Generalization**: Results specific to Titanic dataset; may not apply to other maritime disasters

## Future Enhancements

To improve model performance and robustness, consider:

1. **Advanced Models**
   - Gradient boosting: XGBoost, LightGBM, CatBoost
   - Neural networks with proper regularization
   - Voting/stacking ensembles

2. **Feature Engineering**
   - Extract titles from names (Mr, Mrs, Miss, Master) as proxy for age/status
   - Create interaction features (e.g., FamilySize × Pclass)
   - Engineer features from Cabin (deck level, position)
   - Family survival rate aggregates

3. **Preprocessing Enhancements**
   - Advanced imputation: KNN, MICE, or iterative imputation
   - Target encoding for categorical features
   - Feature selection/dimensionality reduction

4. **Modeling Improvements**
   - Nested cross-validation for robust tuning
   - Class balancing techniques (SMOTE, class weights)
   - Build scikit-learn Pipeline for reproducible workflows
   - Hyperparameter optimization with Bayesian methods

5. **Data Augmentation**
   - Incorporate external data (crew details, survivor lists)
   - Use domain expertise for synthetic feature creation

## Documentation

- **[REPORT.md](REPORT.md)** – Comprehensive technical report with detailed methodology, results, discussion, and recommendations (recommended reading for deeper understanding)
- **test.ipynb** – Fully documented Jupyter notebook with code comments and output
- **test.html** – HTML version of notebook for easy sharing and viewing

## Requirements

All required dependencies are listed in [requirements.txt](requirements.txt). Install with:
```bash
pip install -r requirements.txt
```

Core dependencies:
- **pandas** ≥ 1.0.0 – Data manipulation
- **numpy** ≥ 1.18.0 – Numerical operations
- **scikit-learn** ≥ 0.24.0 – ML models and preprocessing
- **matplotlib** ≥ 3.1.0 – Plotting and visualization
- **seaborn** ≥ 0.10.0 – Statistical graphics
- **jupyter** ≥ 1.0.0 – Notebook environment

## Contributing

Feel free to fork this repository, submit issues, or propose enhancements. This project serves as a learning resource and benchmark for Titanic survival prediction tasks.

## References

- **Dataset**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- **Data Source**: [DataScienceDojo Repository](https://github.com/datasciencedojo/datasets)
- **Scikit-learn Documentation**: [scikit-learn.org](https://scikit-learn.org/)

## License

This project is open source and available for educational and research purposes.

---

**Last Updated**: June 19, 2026  
**Project Status**: Completed
