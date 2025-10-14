
---

## 🔢 Logistic Regression

Despite its name, **Logistic Regression** is used for **classification**, not regression.  
It predicts the **probability** that a given input belongs to a certain class (e.g., spam vs. not spam, disease vs. healthy).

---

### 📘 Concept

The goal is to estimate the probability that a sample belongs to class **1**:

$$
P(y = 1 \mid X) = \sigma(\beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n)
$$

Where the **sigmoid function** $\sigma(z)$ maps any real number into a range between 0 and 1:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

If $P(y=1|X) > 0.5$, the model predicts class **1**, else class **0**.

---

### ⚙️ Objective Function

We can’t use Mean Squared Error here (since output is probability),  
so we use the **Log Loss (Binary Cross-Entropy)**:

$$
J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \Big]
$$

where  
- $m$ — number of samples  
- $\hat{y}^{(i)} = \sigma(\beta^T X^{(i)})$ — predicted probability  
- $y^{(i)}$ — true label (0 or 1)

We minimize this cost using **Gradient Descent**.

---

### 🧩 Gradient Descent Update Rule

$$
\beta_j := \beta_j - \alpha \frac{\partial J}{\partial \beta_j}
$$

where

$$
\frac{\partial J}{\partial \beta_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})x_j^{(i)}
$$

---

### 📊 Example (Binary Classification with Scikit-learn)
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Example dataset (binary classification)
data = {
    "Hours_Studied": [2, 3, 4, 5, 6, 7, 8, 9],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Hours_Studied"]]
y = df["Pass"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
````

---

### 📈 Sigmoid Function Example

```python
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 200)
plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("σ(z)")
plt.grid(True)
plt.show()
```

---

### 🧮 Interpretation

| Metric                       | Meaning                                                |
| ---------------------------- | ------------------------------------------------------ |
| **Coefficients ($\beta_i$)** | Log-odds change in outcome per unit change in `x_i`    |
| **Intercept ($\beta_0$)**    | Baseline log-odds when all features are 0              |
| **Accuracy**                 | Fraction of correctly predicted labels                 |
| **Confusion Matrix**         | Shows TP, FP, FN, TN counts                            |
| **Precision / Recall / F1**  | Evaluates balance between correctness and completeness |

---

### 🧠 Intuition

* Logistic Regression predicts **probabilities**, not direct classes.
* The **decision boundary** is where $P(y=1|X) = 0.5$.
* It is a **linear classifier**, meaning the boundary between classes is a straight line (or hyperplane).

---

### ⚠️ Limitations

| Problem             | Description                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| **Non-linearity**   | Fails on data not linearly separable (unless you add polynomial features) |
| **Outliers**        | Can heavily influence the decision boundary                               |
| **Imbalanced Data** | Tends to favor majority class                                             |

---

### ✅ When to Use

* Binary or multiclass classification
* Predicting probabilities
* Interpretable models (e.g., credit scoring, medical diagnosis)
