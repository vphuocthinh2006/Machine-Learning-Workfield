
---

## 🌀 K-Means Clustering

**K-Means** is an **unsupervised learning algorithm** used to group similar data points into **K distinct clusters**.
It tries to minimize the distance between data points and their assigned cluster centers (called **centroids**).

---

### 🧠 Intuition

Think of K-Means as a way to automatically “label” data by similarity —
each cluster contains points that are close to each other in feature space.

It’s widely used for:

* Market segmentation
* Image compression
* Anomaly detection
* Document grouping

---

### ⚙️ Algorithm Steps

1. **Choose the number of clusters (K)**
   Decide how many groups you want to divide your data into.

2. **Initialize centroids**
   Randomly select K points from the dataset as initial cluster centers.

3. **Assign points to the nearest centroid**
   Each point joins the cluster whose centroid is closest to it (usually via Euclidean distance).

4. **Recalculate centroids**
   Update each centroid as the mean of all points currently in its cluster.

5. **Repeat Steps 3–4**
   Until centroids stop changing (convergence).

---

### 🧮 Objective Function

The algorithm minimizes the **within-cluster sum of squares (WCSS)**, also called **inertia**:

```
J = Σ Σ ||xᵢ - μⱼ||²
```

Where:

* xᵢ = data point
* μⱼ = cluster centroid
* The outer sum runs over all clusters

---

### 🧩 Example (Visualization with sklearn)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic 2D data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

### 📊 Choosing the Right K

You can use the **Elbow Method** to decide how many clusters are optimal:

1. Compute the total WCSS (inertia) for different K values.
2. Plot WCSS vs K.
3. The “elbow point” (where the curve bends) is a good choice for K.

```python
inertia = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    inertia.append(km.inertia_)

plt.plot(range(1, 10), inertia, 'bo-')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.show()
```

---

### 🚀 Summary

| Aspect              | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| **Type**            | Unsupervised                                                |
| **Goal**            | Group similar data points                                   |
| **Distance Metric** | Usually Euclidean                                           |
| **Strengths**       | Simple, scalable, fast                                      |
| **Weaknesses**      | Sensitive to K, initialization, and outliers                |
| **Variants**        | MiniBatch K-Means, K-Medoids, Gaussian Mixture Models (GMM) |

---
