
---

## 🎬 Collaborative Filtering (CF)

**Collaborative Filtering** is a type of **recommender system** that predicts a user’s interests
by learning from the **interactions between users and items** — for example, movie ratings, likes, or purchases.

It’s called *“collaborative”* because it uses the **collective behavior** of many users to make personalized recommendations.

---

### 🧠 Intuition

If two users have **rated similar movies similarly**,
then they likely share **similar preferences** — so one user’s favorite movie can be recommended to the other.

Collaborative Filtering **doesn’t need explicit features** of users or items.
It learns latent factors (hidden patterns) that explain preferences.

---

### ⚙️ The Setup

We start with a matrix ( Y ):

|            | Movie 1 | Movie 2 | Movie 3 | Movie 4 |
| :--------: | :-----: | :-----: | :-----: | :-----: |
| **User 1** |    5    |    ?    |    4    |    ?    |
| **User 2** |    3    |    2    |    ?    |    1    |
| **User 3** |    ?    |    4    |    5    |    ?    |

Goal → Predict the missing entries (“?”).

---

### 🔢 Matrix Factorization Formulation

Collaborative Filtering models the rating matrix ( Y ) as the **product of two low-rank matrices**:

```
Y ≈ X × Wᵀ
```

Where:

* ( X ) → item feature matrix (num_movies × num_features)
* ( W ) → user preference matrix (num_users × num_features)
* ( b ) → optional bias terms (user/movie averages)

The predicted rating:

```
ŷ(i, j) = W_jᵀ X_i + b_j
```

---

### 🧮 Cost Function

We minimize the **mean squared error** over all known ratings:

```
J(X, W, b) = ½ Σ ( (W_jᵀ X_i + b_j - Y(i,j))² ) + (λ/2)(||X||² + ||W||²)
```

where:

* λ → regularization parameter (to avoid overfitting)
* the sum is taken only over the rated (non-missing) entries.

---

### 🔁 Training (Gradient Descent)

At each iteration:

```
X_i := X_i - α ∂J/∂X_i
W_j := W_j - α ∂J/∂W_j
```

Until convergence — when prediction error stabilizes.

---

### 🧩 Example with MovieLens Dataset

```python
import numpy as np
from sklearn.metrics import mean_squared_error
from surprise import SVD, Dataset, Reader

# Load the MovieLens 100k dataset
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

# Use SVD (a matrix factorization-based CF model)
model = SVD()
model.fit(trainset)

# Predict a single user-item rating
uid, iid = str(196), str(302)
pred = model.predict(uid, iid)
print(pred.est)
```

---

### 💡 Practical Notes

| Concept                | Description                          |
| ---------------------- | ------------------------------------ |
| **Cold Start Problem** | New users/items have no history      |
| **Sparsity**           | Most users rate few items            |
| **Biases**             | Some users rate higher/lower overall |
| **Regularization**     | Prevents overfitting in sparse data  |

---

### 🚀 Summary

| Aspect               | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| **Type**             | Recommender System                                         |
| **Goal**             | Predict missing preferences                                |
| **Approach**         | Learn latent user/item features                            |
| **Common Algorithm** | Matrix Factorization (SVD, ALS)                            |
| **Libraries**        | `surprise`, `lightfm`, `implicit`, TensorFlow Recommenders |

---

### 🧩 Visualization Concept (Optional)

If plotted, each user and item can be thought of as **points in the same latent space**:
close points → higher predicted rating.

---
