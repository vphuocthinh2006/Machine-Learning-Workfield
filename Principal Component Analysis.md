
---

## 🧭 Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)** is an **unsupervised dimensionality reduction** technique.
It finds the directions (called **principal components**) that **capture the most variance** in your data — essentially compressing it while preserving its structure.

---

### 🧠 Intuition

Imagine you have data in 3D, but all points roughly lie on a 2D plane.
PCA helps you **rotate and project** that data into a lower-dimensional space (e.g., 2D) **without losing important information**.

It does so by:

1. Finding the directions (axes) where the data varies most.
2. Projecting the data onto those new axes.

This is useful for:

* Visualizing high-dimensional data
* Speeding up ML algorithms
* Removing noise / redundancy

---

### ⚙️ Steps of PCA

1. **Normalize** the dataset

   * Subtract the mean and (optionally) scale to unit variance

2. **Compute the covariance matrix**

   ```
   Σ = (1/m) * XᵀX
   ```

   where `X` is your data matrix (zero-centered).

3. **Compute eigenvectors & eigenvalues**

   * Eigenvectors → directions of maximum variance (principal components)
   * Eigenvalues → amount of variance explained by each component

4. **Sort & select top K eigenvectors**

   * These define the reduced feature space

5. **Project data**

   ```
   Z = X * U_reduce
   ```

   where `U_reduce` contains top K eigenvectors.

---

### 🧮 Objective Function

PCA minimizes the **projection error** between the original data and its lower-dimensional representation:

```
minimize ||X - ZUᵀ||²
```

Equivalent to maximizing the variance of projected data.

---

### 🧩 Example: Dimensionality Reduction with sklearn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load sample dataset
data = load_iris()
X = data.data
y = data.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot reduced 2D data
plt.figure(figsize=(6,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title("PCA Projection of Iris Dataset (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

---

### 📊 Explained Variance Ratio

You can check how much information each component retains:

```python
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

For example:

```
[0.924, 0.053]
```

means the first 2 components capture **97.7%** of total variance.

---

### 🚀 Summary

| Aspect          | Description                                                                     |
| :-------------- | :------------------------------------------------------------------------------ |
| **Type**        | Unsupervised Dimensionality Reduction                                           |
| **Goal**        | Find directions (principal components) capturing maximum variance               |
| **Key Idea**    | Project data into a smaller subspace with minimal information loss              |
| **Common Uses** | Visualization, compression, noise reduction, feature decorrelation              |
| **Limitations** | Linear method only; not ideal for nonlinear manifolds (use Kernel PCA or t-SNE) |

---

### 🔮 PCA in Practice

* Used before clustering or regression to simplify features.
* Often applied for **visualizing** high-dimensional embeddings (like word vectors).
* Can help remove **multicollinearity** in regression problems.

---

