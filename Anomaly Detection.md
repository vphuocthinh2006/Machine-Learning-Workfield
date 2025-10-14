
---

## 🕵️‍♂️ Anomaly Detection

**Anomaly Detection** is used to identify **unusual patterns** or **outliers** in data —
points that differ significantly from the majority.

It’s heavily used in:

* Fraud detection 💳
* Server monitoring 🖥️
* Manufacturing defect detection ⚙️
* Medical diagnostics 🩺

---

### 🧠 Intuition

In normal data, most examples cluster around some expected behavior.
If a new example’s probability of occurrence is **very low**,
we label it as **an anomaly**.

[
p(x) < \varepsilon \implies \text{anomaly}
]

where ( \varepsilon ) is a threshold value.

---

### ⚙️ The Statistical Approach

We model normal data using a **probability distribution**, usually **Gaussian** (Normal):

[
p(x) = \prod_{j=1}^{n} p(x_j; \mu_j, \sigma_j^2)
]

Each feature ( x_j ) is assumed to follow:

[
p(x_j; \mu_j, \sigma_j^2) = \frac{1}{\sqrt{2\pi\sigma_j^2}} , \exp!\left(-\frac{(x_j - \mu_j)^2}{2\sigma_j^2}\right)
]

If the resulting ( p(x) ) is very small, that data point is **anomalous**.

---

### 🧩 Steps to Implement

1. **Fit** the model on normal training data (no anomalies).
2. **Estimate parameters:**
   [
   \mu_j = \frac{1}{m} \sum_{i=1}^m x_j^{(i)}, \quad
   \sigma_j^2 = \frac{1}{m} \sum_{i=1}^m (x_j^{(i)} - \mu_j)^2
   ]
3. **Compute** ( p(x) ) for each example.
4. **Select threshold ( \varepsilon )** using a validation set.
5. **Flag anomalies**: if ( p(x) < \varepsilon ).

---

### 💻 Example (Python)

```python
import numpy as np
from sklearn.metrics import f1_score

# Example dataset (normal and anomalies)
X_train = np.random.normal(0, 1, (1000, 2))      # normal data
X_val = np.vstack([np.random.normal(0, 1, (200, 2)), 
                   np.random.uniform(-6, 6, (20, 2))])  # anomalies
y_val = np.array([0]*200 + [1]*20)  # 0=normal, 1=anomaly

# Estimate mean and variance
mu = X_train.mean(axis=0)
sigma2 = X_train.var(axis=0)

# Compute Gaussian probability
def multivariate_gaussian(X, mu, sigma2):
    n = len(mu)
    cov = np.diag(sigma2)
    X = X - mu
    return (1 / ((2*np.pi)**(n/2) * np.linalg.det(cov)**0.5)) * \
           np.exp(-0.5 * np.sum(X @ np.linalg.inv(cov) * X, axis=1))

p_val = multivariate_gaussian(X_val, mu, sigma2)

# Select best epsilon
best_epsilon, best_f1 = 0, 0
for eps in np.linspace(min(p_val), max(p_val), 1000):
    preds = (p_val < eps).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1, best_epsilon = f1, eps

print("Best epsilon:", best_epsilon)
print("Best F1 score:", best_f1)
```

---

### 📊 Visualization Concept

In 2D data:

* Normal points → cluster near mean (high probability)
* Anomalies → fall far from the cluster (low probability region)

---

### 💡 Strengths & Weaknesses

| ✅ Pros                             | ⚠️ Cons                              |
| ---------------------------------- | ------------------------------------ |
| Great for detecting rare events    | Assumes data is normally distributed |
| Works well with small labeled data | Not good with complex patterns       |
| Simple, interpretable              | Sensitive to scaling and outliers    |

---

### 🚀 Summary

| Aspect              | Description                           |
| ------------------- | ------------------------------------- |
| **Type**            | Unsupervised / Semi-supervised        |
| **Goal**            | Detect abnormal points                |
| **Common Approach** | Gaussian distribution modeling        |
| **Output**          | Anomaly score (probability)           |
| **Libraries**       | `sklearn`, `pyod`, `scikit-multiflow` |

---
