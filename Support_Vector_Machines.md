
## ⚔️ Support Vector Machines (SVM)

**Support Vector Machines** (SVM) are powerful supervised learning algorithms used for **classification** and **regression**.  
They aim to find the **optimal hyperplane** that best separates the data into distinct classes.

---

### 🧠 Core Intuition

SVM tries to **maximize the margin** — the distance between the hyperplane and the nearest data points (called **support vectors**).

A larger margin = a more confident and generalizable classifier.

\[
\text{Decision boundary: } w \cdot x + b = 0
\]

\[
\text{Objective: } \min_{w, b} \frac{1}{2} ||w||^2 \quad \text{s.t. } y_i(w \cdot x_i + b) \ge 1
\]

- \( w \): weight vector (normal to the hyperplane)  
- \( b \): bias term  
- \( y_i \): class label (+1 or -1)

---

### 📈 Visualization Intuition

If your data has two features (2D):

- The **solid line** = decision boundary  
- The **dashed lines** = margin boundaries  
- **Support vectors** = points that lie exactly on the dashed lines — they *define* the hyperplane

🧩 *Only support vectors affect the boundary. Other points don’t matter once margin is fixed.*

---

### 💻 Example (Linear SVM)

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load example dataset
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM with linear kernel
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# Plot decision boundary
w = svm.coef_[0]
b = svm.intercept_[0]
x_points = np.linspace(X[:, 0].min(), X[:, 0].max())
y_points = -(w[0] / w[1]) * x_points - b / w[1]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.plot(x_points, y_points, 'k-')
plt.title("Linear SVM Decision Boundary")
plt.show()
````

---

### 🧮 Soft Margin SVM

Real-world data is **not perfectly separable**.
SVM introduces **slack variables** (ξ) to allow some misclassifications:

[
\min_{w, b, \xi} \frac{1}{2}||w||^2 + C \sum_{i=1}^{n} \xi_i
]
[
\text{s.t. } y_i(w \cdot x_i + b) \ge 1 - \xi_i, ; \xi_i \ge 0
]

* ( C ): Regularization parameter

  * High ( C ) → strict margin (less tolerance for errors, may overfit)
  * Low ( C ) → wider margin (more general, may underfit)

---

### 🌈 Nonlinear SVM and the Kernel Trick

When data **isn’t linearly separable**, SVM uses the **kernel trick** to project data into higher dimensions where a linear separator *can* exist.

Common kernels:

| Kernel         | Formula                | Description              |        |   |       |                                                  |
| -------------- | ---------------------- | ------------------------ | ------ | - | ----- | ------------------------------------------------ |
| Linear         | ( x \cdot x' )         | Simple, fast             |        |   |       |                                                  |
| Polynomial     | ( (x \cdot x' + c)^d ) | Models polynomial curves |        |   |       |                                                  |
| RBF (Gaussian) | ( \exp(-\gamma         |                          | x - x' |   | ^2) ) | Most popular, models smooth nonlinear boundaries |

---

#### 💻 Example (RBF Kernel SVM)

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

# Generate nonlinear data
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

# Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', gamma=0.7, C=1.0)
svm_rbf.fit(X, y)

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300))
Z = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("SVM with RBF Kernel")
plt.show()
```

---

### ⚖️ Comparison: Linear vs Nonlinear

| Model                     | Works On                | Decision Boundary     | Parameters   |
| ------------------------- | ----------------------- | --------------------- | ------------ |
| Linear SVM                | Linearly separable data | Straight line / plane | C            |
| Nonlinear SVM (RBF, Poly) | Nonlinear data          | Curved / flexible     | C, γ, degree |

---

### 🚀 Summary

* **Goal**: Maximize margin between classes
* **Key Concept**: Support vectors define the hyperplane
* **Linear SVM** works for simple data
* **Kernel SVM** handles nonlinear patterns
* **C** controls tradeoff between margin width and misclassification

---

🧭 *Next Section → k-Means Clustering (Unsupervised Learning) 🔵*


