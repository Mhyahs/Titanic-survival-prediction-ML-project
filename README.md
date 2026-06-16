# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using classification models including Logistic Regression, Random Forest, and Decision Tree.

## Overview

This project analyzes the famous Titanic dataset and builds predictive models to determine whether a passenger would survive based on features like class, age, sex, and family relationships. The notebook includes data cleaning, feature engineering, model training, hyperparameter tuning, and comprehensive evaluation.

## Dataset

- **Source**: [DataScienceDojo Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Features**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked (+ engineered FamilySize)
- **Target**: Survived (binary classification)

## Key Results

After hyperparameter tuning with GridSearchCV:
- **Random Forest**: Best performing model with ~80-82% accuracy
- **Logistic Regression**: ~78-80% accuracy
- **Decision Tree**: ~75-78% accuracy

(Exact values depend on the train/test split used when running the notebook)

## Project Structure

```
titanic/
├── test.ipynb          # Main notebook with full analysis
├── test.html           # HTML export of the notebook
├── REPORT.md           # Detailed project report
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── test_files/         # Supporting files
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd titanic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook test.ipynb
```

Or execute the entire notebook and export to HTML:
```bash
jupyter nbconvert --to notebook --execute test.ipynb --inplace
jupyter nbconvert --to html test.ipynb --output test.html
```

## Methodology

### Data Preprocessing
- **Missing values**: Age imputed with median, Embarked with mode, Cabin dropped
- **Encoding**: Sex (male=0, female=1), Embarked (S=0, C=1, Q=2)
- **Scaling**: StandardScaler applied before model training

### Feature Engineering
- Created `FamilySize` = SibSp + Parch + 1 to capture family group size

### Model Training & Tuning
- Baseline models: Logistic Regression, Random Forest, Decision Tree
- Hyperparameter optimization: GridSearchCV with 5-fold cross-validation
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

## Visualizations

The notebook includes:
- Feature importance plot (Random Forest)
- Confusion matrices for all optimized models
- ROC-AUC curves
- Cross-validation score distributions
- Model accuracy comparison chart

## Limitations

- Relatively small historical dataset; results may not generalize
- Simple encoding/imputation strategies used
- Cabin information dropped entirely (could extract deck level)
- Limited feature engineering scope

## Future Improvements

- Advanced feature engineering (title extraction, family structure optimization)
- Try ensemble models (XGBoost, LightGBM, CatBoost)
- Nested cross-validation for robust hyperparameter selection
- Create scikit-learn Pipeline for reproducible preprocessing + modeling
- Advanced imputation techniques (KNN, target encoding)

## Requirements

See [requirements.txt](requirements.txt) for all dependencies.

## Author

Created as a project-based machine learning exercise

## License

Open source - feel free to use and modify
