# deforestation-svm-analysis
Predicting deforestation with Support Vector Regression: kernel comparison, nested cross-validation and feature importance analysis
# Deforestation Issue Analysis Using Support Vector Machine (SVM)

Analyzing deforestation across countries with **Support Vector Regression (SVR)**: comparing kernels, tuning hyperparameters with cross-validation, and identifying which factors (population, GDP, rainfall, policy strictness, illegal lumbering, etc.) are linked to forest loss.

## Project Workflow

**Phase 1 – Data Preprocessing**
- Data inspection, missing values and duplicates
- Encoding of categorical columns
- Standardization of features and target inside the model pipeline (fitted on training data only)
- 80/20 train-test split
- Feature selection check using cross-validation

**Phase 2 – Model Building and Evaluation**
- Linear-kernel SVR as the initial model (train and test performance)
- Polynomial and RBF kernels
- `GridSearchCV` over kernel, C, gamma, epsilon and polynomial degree
- Nested cross-validation (tuning repeated inside each fold) and a predict-the-mean baseline
- Metrics: MAE, MSE, RMSE, R²

**Phase 3 – Feature Analysis and Interpretation**
- Permutation importance, linear SVR weights and Spearman correlations (with multiple-testing correction)
- Feature-by-feature notes on GDP, population, rainfall, policy strictness, corruption and illegal lumbering
- Group averages and country rankings

**Phase 4 – Reporting and Recommendations**
- Heatmap, importance chart, actual vs predicted, kernel comparison, scatter plots, partial-dependence plots and country charts
- Generated summary and recommendations

## Method Notes

- **Target:** `Forest_Loss_Area_km2` (the other loss column, `Tree_Cover_Loss_percent`, is excluded from the inputs because it measures the same outcome).
- The target is scaled inside the model: SVR with an unscaled target of this size predicts a near-constant value.
- Country is used for reporting only; label codes have no numeric meaning.
- Decision boundaries do not apply to regression, so feature importance and partial dependence are used instead.
- A model with R² at or below 0 is no better than always predicting the average.

## Results

*(Fill in after running the script on the real dataset.)*

| Model | Test MAE | Test RMSE | Test R² | CV R² (train) |
|---|---|---|---|---|
| Mean baseline | – | – | – | – |
| Linear SVR | – | – | – | – |
| Polynomial SVR | – | – | – | – |
| RBF SVR | – | – | – | – |
| Tuned SVR | – | – | – | – |

Nested CV R² (tuned SVR vs. mean baseline): – vs –

Top features: –

## Limitations

- Only about 100 records, so results vary a lot between splits and country-level rankings rest on very few records per country.
- Associations are not causal.
- If the model does not beat the baseline, its feature importances are not reliable evidence about the drivers of deforestation.

## Dataset

`deforestation_dataset.csv` should be placed in the project root next to the notebook.
Download link: https://drive.google.com/file/d/1q2lCwk-EmQ7kIImImXOn3iEzEUkgZcBA/view?usp=sharing



## Tech Stack

Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn
