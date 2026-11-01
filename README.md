# Strength Prediction with Linear Regression

A machine learning project that predicts material **strength** from ingredient composition and age using **Linear Regression**.

---

## 📌 Overview

This project builds a regression model that learns the relationship between a material's **ingredients** (and **age**) and its resulting **strength**. Given new ingredient amounts and an age, the model predicts the expected strength.

The workflow is a clean, minimal example of a supervised learning pipeline:

1. Separate features (`X`) from the target (`y`)
2. Split data into training and test sets
3. Train a Linear Regression model
4. Evaluate predictions on unseen data

---

## 🎯 Problem Statement

**Goal:** Predict the **strength** of a material based on:
- The **ingredients** used (composition)
- The **age** of the material

**Target variable:** `strength`

**Input features:** Everything except `strength` (all ingredients + age)

---

## 📂 Dataset

The dataset is loaded into a pandas DataFrame `df` with the following structure:

| Column | Description |
|--------|-------------|
| `ingredient_1` … `ingredient_n` | Amounts of each ingredient |
| `age` | Age of the material |
| `strength` | **Target** — the value to predict |

> ⚠️ Update this table with your actual column names and units (e.g., grams, days, MPa).

---

## 🛠️ Requirements

```
python >= 3.8
pandas
numpy
scikit-learn
matplotlib        # optional, for visualization
jupyter           # optional, for notebooks
```

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## 🚀 Usage

### 1. Load the data

```python
import pandas as pd

df = pd.read_csv("data.csv")   # replace with your file
df.head()
```

### 2. Separate features and target

```python
# X = all ingredients and age (everything except strength)
X = df.drop('strength', axis=1)

# y = the target we want to predict (strength)
y = df['strength']
```

### 3. Split into training and test sets

```python
from sklearn.model_selection import train_test_split

# random_state=42 ensures you get the exact same split every time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- **80%** of the data is used for training
- **20%** is held out for testing
- `random_state=42` makes the split **reproducible**

### 4. Train the model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### 5. Evaluate the model

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

predictions = model.predict(X_test)

mae  = mean_absolute_error(y_test, predictions)
mse  = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2   = r2_score(y_test, predictions)

print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")
```

### 6. Inspect learned coefficients

```python
import pandas as pd

coefs = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
}).sort_values("coefficient", ascending=False)

print("Intercept:", model.intercept_)
print(coefs)
```

This tells you **how much each ingredient (and age) affects strength**.

---

## 📊 Evaluation Metrics

| Metric | Meaning | Better When |
|--------|---------|-------------|
| **MAE** | Average absolute error | Lower |
| **RMSE** | Error in same units as target, penalizes big misses | Lower |
| **R²** | Fraction of variance explained (0–1) | Higher (closer to 1) |

> A model that simply predicts the mean would give **R² = 0**. Positive R² means the model learned something useful.

---

## 🧠 How It Works (Theory)

Linear Regression assumes the target is a **weighted sum** of the inputs plus a constant:

$$
\text{strength} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

- **`w`** = weights (how much each ingredient matters)
- **`b`** = bias (baseline strength)

During `model.fit()`, the model finds the weights that **minimize the sum of squared errors** between predictions and actual strengths — the classic **Ordinary Least Squares (OLS)** solution.

---

## 📁 Project Structure

```
├── data.csv               # dataset
├── notebook.ipynb         # exploratory analysis + modeling
├── train.py               # training script
├── requirements.txt       # dependencies
└── README.md              # this file
```

---

## 💡 Tips & Notes

- **Only test once.** Don't tune the model based on `X_test` results — that leaks information. Use cross-validation for tuning.
- **Feature scaling** isn't required for plain Linear Regression, but helps if you compare with regularized models (Ridge, Lasso).
- **Check assumptions.** Linear Regression assumes linearity, independence of errors, and homoscedasticity (constant error variance). Plot residuals to verify.
- **Watch for multicollinearity.** Highly correlated ingredients can make coefficients unstable.
- **Improve the model** by trying:
  - Feature engineering (ratios, interactions)
  - Polynomial features
  - Ridge / Lasso regularization
  - Tree-based models (Random Forest, Gradient Boosting)

---

## 🔮 Next Steps

- [ ] Add cross-validation
- [ ] Compare with Ridge/Lasso and tree-based models
- [ ] Visualize residual plots and predictions vs. actual
- [ ] Build a simple prediction CLI or web app

---

## 📄 License

Specify your license here (e.g., MIT, Apache 2.0).

---

## 🙏 Acknowledgements

Built with [scikit-learn](https://scikit-learn.org/) and [pandas](https://pandas.pydata.org/).

---

## ✅ Quick Reference

```python
# The whole pipeline in 10 lines
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")
X, y = df.drop('strength', axis=1), df['strength']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
print("R²:", model.score(X_test, y_test))
```

> **In one line:** This project learns a linear formula that predicts material strength from ingredients and age — and tests it fairly on data the model has never seen.