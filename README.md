# 🛒 Retail Price Optimization Using Machine Learning

This project focuses on analyzing retail sales data to build a regression model that predicts optimal product prices. It includes data cleaning, exploratory analysis, feature engineering, and multiple machine learning models, ranging from linear regressors to advanced ensemble techniques.

---

## 📌 Project Overview

- **Goal**: Predict the `lag_price` (historical price) for retail products using historical sales and product metadata.
- **Data Source**: Structured CSV data from Kaggle (uploaded as `retail_price.csv`).
- **Techniques**:
  - Exploratory Data Analysis (EDA)
  - Outlier treatment
  - Time-based and category-based feature engineering
  - Correlation analysis
  - Model training with hyperparameter tuning (RandomizedSearchCV) and early stopping on CatBoost

---

## 🧰 Libraries Used

- **Data Manipulation**: `pandas`, `numpy`, `os`
- **Visualization**: `matplotlib`, `seaborn`, `plotly.express`
- **Modeling**:
  - Traditional: `Ridge`, `Lasso`, `ElasticNet`, `KNN`, `SVR`, `DecisionTree`, `MLP`
  - Ensembles: `RandomForest`, `GradientBoosting`, `AdaBoost`
  - Advanced: `XGBoost`, `CatBoost`, `LightGBM`
- **Utilities**: `scikit-learn`, `warnings`, `datetime`

---

## 📊 Exploratory Data Analysis

Key steps include:
- Distribution analysis of numeric features
- Revenue-based product sorting
- Missing value checks
- Correlation matrix heatmap
- Time series trends (monthly/daily)

> ✅ Improved correlation heatmap with larger size, better label rotation, and clearer layout.

---

## 🧹 Data Preprocessing

- **Outlier Detection & Capping** using IQR method for key features (`qty`, `customers`, `comp_1`, `comp_3`)
- **Datetime Features** parsed from `month_year` to extract `day`
- **One-hot Encoding** applied to `product_category_name`

---

## 🧠 Modeling Pipeline

1. **Feature Matrix & Target Split**:
   - Features (`X`) exclude `lag_price`
   - Target (`y`) is `lag_price`

2. **Train-Test Split**: 70/30 split with `random_state=42`

3. **Models Trained**:
    3. **Models Trained**:
    - ✅ `Ridge`, `Lasso`, `ElasticNet`
    - ✅ `KNeighborsRegressor`, `SVR`
    - ✅ `DecisionTree`, `MLPRegressor`
    - ✅ `RandomForest`, `GradientBoosting`, `AdaBoost`
    - ✅ `XGBoost`, `LightGBM`
    - ✅ **CatBoost (tuned via RandomizedSearchCV with early stopping)**

4. 4. **Evaluation Metric**:
   - **Test set** performance:
     - R² = **0.9701**
     - MSE = **186.69**

---

## 🖼️ Visualizations

- Histograms and KDEs for integer features
- Time-based plots for lag_price trends
- Feature-pair scatterplots (`pairplot`)
- Clean correlation heatmap with `.2f` precision annotations

---

## 🚀 Getting Started

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
catboost
lightgbm
```

### Run the Code

```bash
python main.py
```

Make sure `retail_price.csv` is in the correct path, or update the dataset path inside the script.

---

## 📈 Results

All models are benchmarked using training R² scores for initial assessment. Best models can be fine-tuned via `GridSearchCV` or `RandomizedSearchCV` (already imported).
LGBMRegressor: Train Score = 0.9936 and 
CatBoostRegressor: Train Score = 0.9996
---

## 📂 Project Structure

```
.
├── main.py
├── README.md
└── requirements.txt
```

---

## 🧠 Future Improvements

- Model interpretability with SHAP values
- Pipeline packaging with `scikit-learn` `Pipeline`
- Hyperparameter tuning and cross-validation
- Model deployment using `Flask` or `Streamlit`

---

## 👤 Author

Built with ❤️ by Abhinav
