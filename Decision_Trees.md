
---

## 🌳 Decision Trees

A **Decision Tree** is a supervised learning algorithm used for **classification** and **regression**.  
It works by **recursively splitting the dataset** into smaller subsets based on feature values, forming a tree structure.

---

### 🧠 Intuition

A decision tree mimics **human decision-making** —  
it asks a series of *yes/no questions* about the features to reach a conclusion.

Example:

```

Is temperature > 30°C?
├── Yes → Go swimming
└── No → Is it raining?
├── Yes → Stay home
└── No → Go hiking

````

Each **internal node** = a feature-based question  
Each **leaf node** = a final decision (prediction)

---

### 🧩 How It Works

1. **Choose the best feature to split** the data (based on a criterion)
2. **Divide** the dataset according to that feature’s value  
3. **Repeat** the process recursively for each subset until:
   - All samples belong to one class, or  
   - The maximum depth is reached

---

### 📐 Mathematical Foundation

Decision trees aim to **maximize purity** (reduce uncertainty) in each split.  
Common metrics:

#### 🔹 Entropy (Information Gain)

\[
Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
\]

\[
InformationGain = Entropy(parent) - \sum_{k} \frac{|S_k|}{|S|} Entropy(S_k)
\]

#### 🔹 Gini Impurity

\[
Gini(S) = 1 - \sum_{i=1}^{c} p_i^2
\]

Both measure how “mixed” the data is —  
lower impurity = more homogeneous node.

---

### 🧮 Example Calculation

Suppose a node has:
- 3 samples of Class A  
- 1 sample of Class B  

Then:
\[
p_A = \frac{3}{4}, \quad p_B = \frac{1}{4}
\]
\[
Entropy = -\left(\frac{3}{4}\log_2\frac{3}{4} + \frac{1}{4}\log_2\frac{1}{4}\right) = 0.811
\]
\[
Gini = 1 - \left[\left(\frac{3}{4}\right)^2 + \left(\frac{1}{4}\right)^2\right] = 0.375
\]

---

### 💻 Example (Scikit-learn)

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf.fit(X, y)

# Plot
plt.figure(figsize=(12,6))
plot_tree(clf, filled=True, feature_names=load_iris().feature_names, class_names=load_iris().target_names)
plt.title("Decision Tree - Iris Dataset")
plt.show()
````

---

### 🌈 Visualization Intuition

* Each **split** = a boundary in feature space
* The tree **divides** space into regions
* Each **region** = one predicted class

Decision Trees are **piecewise linear** models —
the decision boundaries are **axis-aligned rectangles** (in 2D).

---

### ⚙️ Hyperparameters

| Parameter           | Description                             |
| ------------------- | --------------------------------------- |
| `criterion`         | Split metric (`entropy`, `gini`)        |
| `max_depth`         | Maximum depth of tree                   |
| `min_samples_split` | Minimum samples needed to split a node  |
| `min_samples_leaf`  | Minimum samples in a leaf node          |
| `max_features`      | Number of features considered per split |
| `random_state`      | Reproducibility seed                    |

---

### ⚖️ Pros and Cons

| ✅ Advantages                             | ⚠️ Disadvantages                            |
| ---------------------------------------- | ------------------------------------------- |
| Easy to interpret                        | Can overfit easily                          |
| No need for scaling                      | Sensitive to small data changes             |
| Handles categorical + numerical features | Can create biased splits on unbalanced data |
| Works for both classification/regression | Less stable than ensemble methods           |

---

### 🚀 Summary

* Decision Trees split data recursively to **maximize purity**
* They are **intuitive, interpretable, and flexible**
* Prone to **overfitting**, but powerful when combined into **ensembles** like Random Forest or Gradient Boosting

