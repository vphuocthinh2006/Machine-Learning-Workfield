
---

## 🎯 Content-Based Filtering (CBF)

**Content-Based Filtering** recommends items that are **similar** to what the user has liked before —
based on the *content* (attributes/features) of the items themselves.

Unlike **Collaborative Filtering**, it doesn’t rely on other users’ behavior.
It uses **item features** and **user profiles** to make predictions.

---

### 🧠 Intuition

If you liked *Inception* (a sci-fi, thriller movie),
the system recommends other movies with **similar genres, actors, or descriptions**.

Each item is described by a **feature vector**, and each user has a **preference vector** learned from their past ratings.

---

### ⚙️ Core Idea

1. Represent every item ( i ) by a **feature vector** ( x^{(i)} )
   (e.g., genre, keywords, year, etc.)
2. Represent each user ( j ) by a **weight vector** ( w^{(j)} )
   (capturing their preference importance for each feature)
3. Predict the rating for user ( j ) on item ( i ):

[
\hat{y}^{(i,j)} = (w^{(j)})^T x^{(i)}
]

---

### 🧩 Example Workflow

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Example movie dataset
data = {
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Avengers'],
    'genre': ['sci-fi thriller', 'sci-fi drama', 'action crime', 'action fantasy']
}

df = pd.DataFrame(data)

# Step 1: Encode item content using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['genre'])

# Step 2: Compute item-to-item similarity
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 3: Recommend movies similar to "Inception"
movie_index = df[df.title == 'Inception'].index[0]
similar_scores = list(enumerate(similarity[movie_index]))
sorted_movies = sorted(similar_scores, key=lambda x: x[1], reverse=True)[1:]

print("Movies similar to Inception:")
for idx, score in sorted_movies:
    print(f"{df.iloc[idx]['title']} (score={score:.2f})")
```

---

### 📘 Formula Recap

[
\text{Similarity}(x_i, x_j) = \frac{x_i \cdot x_j}{||x_i|| , ||x_j||}
]

This is **cosine similarity**, which measures the angle between two feature vectors.

* **1.0** → perfectly similar
* **0.0** → completely different

---

### 💡 Strengths & Weaknesses

| ✅ Pros                                 | ⚠️ Cons                             |
| :------------------------------------- | :---------------------------------- |
| No need for other users’ data          | Cannot suggest *novel* items        |
| Works well for new items with features | Feature extraction can be hard      |
| Personalized for each user             | Narrow — only similar to past likes |

---

### 🧩 Hybrid Systems (Bonus)

In practice, most modern recommendation systems (e.g., Netflix, Spotify) combine:

* **Collaborative Filtering (CF)** → learns from others’ behavior
* **Content-Based Filtering (CBF)** → uses item metadata
* **Hybrid Models** → blend both to overcome cold-start/sparsity

---

### 🚀 Summary

| Aspect           | Description                                             |
| ---------------- | ------------------------------------------------------- |
| **Type**         | Recommender System                                      |
| **Goal**         | Recommend similar items based on content                |
| **Approach**     | Cosine similarity or linear regression on item features |
| **Common Tools** | `sklearn`, `TF-IDF`, `cosine_similarity`                |
| **Used in**      | Netflix, Spotify, Amazon (early systems)                |

---
