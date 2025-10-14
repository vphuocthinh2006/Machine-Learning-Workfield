
## 🌲 Tree Ensembles (Random Forest, Gradient Boosting, XGBoost)

Decision Trees are simple yet powerful — but on their own, they often **overfit**.  
Tree Ensembles solve this by combining **multiple trees** into a **stronger model**.

Ensemble = “wisdom of the crowd.”  
Each tree votes, and together they make a more robust decision.

---

### 🧩 1. Random Forest

A **Random Forest** builds **many decision trees** on **random subsets** of data and features,  
then averages (for regression) or votes (for classification) their results.

#### ⚙️ How It Works

1. **Bootstrap Sampling** — randomly sample data *with replacement*  
2. **Random Feature Selection** — each tree sees only a subset of features  
3. **Train many trees** independently  
4. **Aggregate predictions**
   - Classification → Majority vote  
   - Regression → Mean of predictions

\[
\hat{y} = \frac{1}{T}\sum_{t=1}^{T} f_t(x)
\]

where \( f_t(x) \) is the prediction from tree \( t \).

#### 💻 Example (Scikit-learn)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
````

#### ✅ Pros

* Reduces overfitting dramatically
* Handles high-dimensional data well
* Works for both classification & regression

#### ⚠️ Cons

* Less interpretable than a single tree
* More computationally expensive

---

### ⚡ 2. Gradient Boosting

Unlike Random Forest (parallel trees), **Gradient Boosting builds trees sequentially** —
each new tree corrects the errors of the previous one.

#### 🧠 Concept

Each tree tries to fit the **residuals** (the errors) from the previous model.

[
y = f_1(x) + f_2(x) + \dots + f_M(x)
]
[
f_{m}(x) = f_{m-1}(x) + \eta \cdot h_m(x)
]

where:

* ( h_m(x) ): new weak learner (a small tree)
* ( \eta ): learning rate (controls update strength)

Each iteration minimizes the **loss function** via gradient descent in function space.

---

#### 💻 Example (GradientBoostingClassifier)

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Synthetic data
X, y = make_classification(n_samples=1000, n_features=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)

# Evaluate
y_pred = gb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

### 🔥 3. XGBoost (Extreme Gradient Boosting)

**XGBoost** is an optimized version of Gradient Boosting —
it’s faster, regularized, and designed for large-scale machine learning.

#### ⚙️ Key Innovations

| Feature                | Description                                                     |
| ---------------------- | --------------------------------------------------------------- |
| Regularization         | Prevents overfitting via L1 (Lasso) and L2 (Ridge) penalties    |
| Parallelization        | Uses multi-threaded CPU cores for faster training               |
| Missing Value Handling | Smartly learns which branch missing values should go            |
| Tree Pruning           | Uses “max depth” instead of “max leaf nodes” for better control |
| Cache Awareness        | Optimized for hardware efficiency                               |

---

#### 🧮 Objective Function

[
Obj = \sum_{i=1}^{n} l(y_i, \hat{y}*i) + \sum*{k=1}^{K} \Omega(f_k)
]

[
\Omega(f) = \gamma T + \frac{1}{2}\lambda ||w||^2
]

where:

* ( l ): loss function (e.g., logistic loss)
* ( \Omega(f) ): regularization term
* ( T ): number of leaves
* ( w ): leaf weights

---

#### 💻 Example (XGBoost)

```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train XGBoost
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8)
xgb.fit(X_train, y_train)

# Evaluate
y_pred = xgb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

### ⚖️ Comparison Summary

| Model             | Approach                 | Bias   | Variance | Speed       | Regularization |
| ----------------- | ------------------------ | ------ | -------- | ----------- | -------------- |
| Decision Tree     | Single tree              | Low    | High     | ✅ Fast      | ❌ None         |
| Random Forest     | Parallel trees (bagging) | Medium | Low      | ⚙️ Medium   | ⚪ Implicit     |
| Gradient Boosting | Sequential trees         | Low    | Medium   | ⏳ Slow      | ✅ Partial      |
| XGBoost           | Optimized boosting       | Low    | Low      | ⚡ Very Fast | ✅ Strong       |

---

### 🚀 Summary

* **Random Forest**: Great baseline; ensemble of uncorrelated trees (bagging)
* **Gradient Boosting**: Sequential, focuses on errors (boosting)
* **XGBoost**: Industrial-strength gradient boosting; fast, regularized, powerful

---
