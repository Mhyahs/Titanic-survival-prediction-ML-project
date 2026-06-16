# Titanic Survival Prediction — Project Report

## Overview
This project builds and evaluates machine learning models to predict passenger survival on the Titanic using the Kaggle / DataScienceDojo Titanic dataset. The workflow includes data cleaning, feature engineering, model training (Logistic Regression, Random Forest, Decision Tree), hyperparameter tuning, and visual evaluation.

## Dataset
- Source: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
- Key features used: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, and engineered `FamilySize`.

## Preprocessing
- Missing values:
  - `Age` imputed with median
  - `Embarked` imputed with mode
  - `Cabin` column dropped (too many missing values)
- Categorical encoding:
  - `Sex`: male=0, female=1
  - `Embarked`: S=0, C=1, Q=2
- Identifiers removed: `PassengerId`, `Name`, `Ticket`.
- Features scaled with `StandardScaler` before training most models.

## Feature Engineering
- `FamilySize` = `SibSp` + `Parch` + 1 (captures family group size on board).
- Features re-prepared and re-scaled after adding `FamilySize`.

## Models & Training
- Baseline models trained:
  - Logistic Regression (sklearn)
  - Random Forest Classifier
  - Decision Tree Classifier
- Hyperparameter tuning performed using `GridSearchCV` (5-fold CV, `n_jobs=-1`, `verbose=0`) for each model to find best parameters.

## Evaluation Metrics & Visualization
- Primary metric: Accuracy (reported on hold-out test set).
- Additional metrics: Precision, Recall, F1-score (classification report), ROC-AUC, Confusion Matrices.
- Cross-validation score distributions visualized with boxplots.
- Feature importance extracted from Random Forest and plotted as a horizontal bar chart.

## Key Findings
- All preprocessing and modeling steps are implemented in the notebook `test.ipynb`.
- Optimized model accuracies are printed and visualized in the notebook; the exact values depend on the train/test split executed in the notebook run.
- Random Forest typically yields higher feature importance stability and competitive accuracy for this dataset.

## Limitations
- The dataset is relatively small and historical; results may not generalize beyond this dataset.
- Simple encoding and imputation strategies were used; more advanced techniques (e.g., KNN imputation, target encoding, or embeddings) might improve performance.
- Cabin information was dropped due to many missing values — more careful imputation or feature extraction (deck level) could help.

## Suggested Next Steps
- Engineer additional features (title extraction from `Name`, better family structure features).
- Try ensemble or boosting models (XGBoost, LightGBM, CatBoost) and compare with Random Forest.
- Use nested cross-validation for more robust hyperparameter selection.
- Create a small pipeline (scikit-learn `Pipeline`) to streamline preprocessing + modeling and make the notebook reproducible.

## How to reproduce and export
From the project folder, you can run these commands (if you have Jupyter and nbconvert installed):

```bash
# Run the notebook to regenerate outputs
jupyter nbconvert --to notebook --execute test.ipynb --inplace

# Export to HTML
jupyter nbconvert --to html test.ipynb --output test.html

# Export to PDF (requires LaTeX)
jupyter nbconvert --to pdf test.ipynb --output test.pdf

# Export to Markdown or Python script
jupyter nbconvert --to markdown test.ipynb
jupyter nbconvert --to script test.ipynb
```

## Files
- Notebook: [test.ipynb](test.ipynb)
- HTML export (generated): [test.html](test.html)
- This report: [REPORT.md](REPORT.md)

## Contact / Notes
- The notebook contains both Persian and English explanatory sections; English comment titles were added to many code cells to improve navigation. Continue finishing titles for all remaining cells for consistency.

---
Report generated from the current notebook contents in the repository. For exact numeric results (accuracies, AUCs, feature importance values), execute `test.ipynb` and review the printed outputs and plots.