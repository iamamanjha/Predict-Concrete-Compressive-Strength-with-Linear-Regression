# 🏗️ Concrete Strength Prediction with Linear Regression

A beginner-friendly machine learning project that predicts **concrete compressive strength (MPa)** from its ingredient composition and curing age using **Linear Regression**, built with scikit-learn.

---

## 📌 Overview

This project demonstrates a complete, end-to-end **supervised regression** pipeline on the **Concrete Compressive Strength** dataset. It walks through:

1. Loading the dataset from a public URL
2. Exploring the data with summary statistics
3. Separating features (`X`) from the target (`y`)
4. Splitting data into training and test sets
5. Training a Linear Regression model
6. Evaluating performance with standard regression metrics
7. Interpreting the learned coefficients (weights)
8. Making predictions on custom inputs

---

## 🎯 Problem Statement

**Goal:** Predict the **compressive strength** of concrete (in **MPa**) based on its **ingredient composition** and **curing age**.

**Target variable:** `strength` — concrete compressive strength (MPa)

**Input features:** 8 features — cement, slag, ash, water, superplasticizer, coarse aggregate, fine aggregate, and age.

---

## 📂 Dataset

The dataset is loaded directly from a public GitHub URL:

```python
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"
df = pd.read_csv(url)
```

- **Size:** 1,030 rows × 9 columns
- **Source:** [UCI Concrete Compressive Strength Dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) (via the *Machine Learning with R* repo)

### Feature Description

| Feature | Description | Unit |
|---------|-------------|------|
| `cement` | Cement content | kg/m³ |
| `slag` | Blast furnace slag | kg/m³ |
| `ash` | Fly ash | kg/m³ |
| `water` | Water content | kg/m³ |
| `superplastic` | Superplasticizer | kg/m³ |
| `coarseagg` | Coarse aggregate | kg/m³ |
| `fineagg` | Fine aggregate | kg/m³ |
| `age` | Curing age | days |
| **`strength`** | **Compressive strength (target)** | **MPa** |

---

## 🛠️ Requirements

```
python >= 3.8
pandas
scikit-learn
jupyter    # optional, for running the notebook
```

Install with:

```bash
pip install pandas scikit-learn jupyter
```

---

## 🚀 Usage

### 1. Load and explore the dataset

```python
import pandas as pd

url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"
df = pd.read_csv(url)

df.head()
df.describe()
```

### 2. Separate features and target

```python
# X = all ingredients and age (everything except strength)
X = df.drop('strength', axis=1)

# y = target (compressive strength)
y = df['strength']
```

### 3. Split into training and test sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- **80%** of rows for training, **20%** for testing
- `random_state=42` makes the split reproducible

### 4. Train the model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### 5. Make predictions and evaluate

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
```

---

## 📊 Results

The model trained with `random_state=42` and an 80/20 train-test split achieved:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MSE** | `95.97` | Average squared error (MPa²) |
| **R²** | `0.6276` | The model explains ~**62.8%** of the variance in concrete strength |

> An R² of ~0.63 is a reasonable baseline for **plain Linear Regression** on this dataset. Non-linear models (Random Forest, Gradient Boosting) can reach R² ≈ 0.90+.

---

## 🧠 Model Interpretation

The learned weights show **how each feature affects concrete strength**, holding all others constant:

```python
weights = pd.DataFrame(model.coef_, X.columns, columns=['Weight'])
```

| Feature | Weight | Effect on Strength |
|---------|--------|--------------------|
| `superplastic` | **+0.3106** | 🥇 Strongest positive influence (per kg/m³) |
| `cement` | +0.1190 | More cement → stronger concrete |
| `age` | +0.1164 | Concrete strengthens as it cures |
| `slag` | +0.1085 | Positive contribution |
| `ash` | +0.0822 | Small positive contribution |
| `fineagg` | +0.0241 | Small positive effect |
| `coarseagg` | +0.0179 | Small positive effect |
| `water` | **−0.1353** | 🥈 More water → **weaker** concrete |

**Intercept:** `−28.82`

> 💡 **Note:** Coefficients are on very different scales because features have different units and ranges. To compare feature importance fairly, standardize the features first (`StandardScaler`).

---

## 🔮 Predicting on Custom Input

```python
new_sample = pd.DataFrame({
    'cement': [300],
    'slag': [50],
    'ash': [0],
    'water': [180],
    'superplastic': [5],
    'coarseagg': [1000],
    'fineagg': [700],
    'age': [28]
})

predicted_strength = model.predict(new_sample)
print("Predicted Strength (MPa):", predicted_strength[0])
# → 27.50 MPa
```

The model predicts a compressive strength of about **27.5 MPa** for this mix after 28 days of curing.

---

## 📁 Project Structure

```
├── notebook1.ipynb      # Jupyter notebook with the full pipeline
├── README.md            # this file
└── requirements.txt     # dependencies (optional)
```

---

## 📈 Evaluation Metrics Explained

| Metric | What It Measures | Better When |
|--------|------------------|-------------|
| **MSE** | Mean Squared Error — average squared difference between predictions and truth | Lower |
| **R²** | Fraction of variance in the target explained by the model (0 = no better than predicting the mean, 1 = perfect) | Higher (closer to 1) |

> ℹ️ **MAE** and **RMSE** were not computed in this notebook, but they're easy to add and are often easier to interpret than MSE since they're in MPa.

```python
from sklearn.metrics import mean_absolute_error
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MAE:  {mae:.2f} MPa")
print(f"RMSE: {rmse:.2f} MPa")
```

---

## 💡 Key Takeaways

- ✅ A complete supervised regression workflow in ~10 lines of code
- ✅ Linear Regression provides a **fast, interpretable baseline**
- ✅ **Water reduces strength**; **cement, superplasticizer, and age increase it**
- ⚠️ The model assumes **linear relationships** — concrete strength is known to be highly non-linear
- ⚠️ **Feature scaling** matters here: features span very different ranges (age up to 365 days, aggregates ~1000 kg/m³), so raw coefficients are not directly comparable

---

## 🚀 Ideas for Improvement

- [ ] **Feature scaling** (`StandardScaler`) — makes coefficients comparable and helps regularized models
- [ ] **Regularization** — try `Ridge`, `Lasso`, or `ElasticNet`
- [ ] **Polynomial features** — capture non-linear behavior (e.g., `age` has diminishing returns)
- [ ] **Tree-based models** — `RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`
- [ ] **Cross-validation** — more robust performance estimate (`cross_val_score`)
- [ ] **Residual analysis** — check linearity, homoscedasticity, and normality of errors
- [ ] **Feature engineering** — water/cement ratio is a well-known predictor of strength
- [ ] **Log-transform the target** — `strength` is skewed; a log transform may improve fit

---

## ⚠️ Notes & Caveats

- The test set was used **once**, at the end, for a fair estimate of performance.
- `random_state=42` guarantees the **same split every run** — essential for reproducibility.
- Do **not** tune the model based on test-set scores; use a **validation set** or cross-validation instead.
- Linear Regression assumes a **linear additive relationship** between each feature and strength — a simplification for this dataset, but a useful baseline.
- Coefficients can be misleading without feature scaling, since units and ranges differ widely.

---

## 📄 License

Specify your preferred license (e.g., MIT, Apache 2.0).

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) — models and metrics
- [pandas](https://pandas.pydata.org/) — data handling
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) — original Concrete Compressive Strength dataset (Yeh, 1998)

---

## ✅ Quick Reference (Full Pipeline)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"
df = pd.read_csv(url)

# 2. Split features / target
X, y = df.drop('strength', axis=1), df['strength']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train
model = LinearRegression().fit(X_train, y_train)

# 5. Predict & evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² :", r2_score(y_test, y_pred))
```

> **One-line summary:** A clean Linear Regression baseline that predicts concrete compressive strength from 8 mixture and curing features, achieving **R² ≈ 0.63** — a solid starting point before moving to more powerful non-linear models.