
# ALS Movie Recommendation System

This project implements a **Movie Recommendation System** using the **Alternating Least Squares (ALS)** algorithm — a powerful collaborative filtering technique that powers platforms like **Netflix**, **Spotify**, and **YouTube**.

The goal is to predict what movies a user might like based on their past ratings and the preferences of similar users.

---

## Table of Contents

1. [Overview](#-overview)
2. [What is ALS?](#-what-is-als)
3. [How It Works](#-how-it-works)
4. [Example Walkthrough](#-example-walkthrough)
5. [Why “Alternating Least Squares”?](#-why-alternating-least-squares)
6. [Key Terms](#-key-terms)
7. [Results](#-results)
8. [How to Run](#-how-to-run)
9. [Real-World Applications](#-real-world-applications)
10. [References](#-references)

---

## Overview

This project demonstrates how **ALS matrix factorization** can predict unseen movie ratings by discovering **hidden patterns** in user–movie data.

We use the **MovieLens 100K dataset** — containing 100,000 real ratings given by 943 users to 1,682 movies.

---

## 🧠 What is ALS?

**ALS (Alternating Least Squares)** is a **collaborative filtering** algorithm that predicts missing ratings in a large, sparse user–item matrix.

Imagine a giant table:
- Rows → represent users  
- Columns → represent movies  
- Cells → represent ratings (1–5 stars)

Most cells are **empty** because users haven’t rated most movies.  
ALS helps us **fill in those blanks** intelligently.

---

## ⚙️ How It Works

### Step 1: Input Data
The dataset looks like this:

| user_id | movie_id | rating | timestamp  |
|----------|-----------|---------|-------------|
| 196      | 242       | 3       | 881250949   |
| 186      | 302       | 3       | 891717742   |
| 22       | 377       | 1       | 878887116   |

---

### Step 2: Matrix Factorization
ALS breaks the big user–movie rating matrix into two smaller matrices:

- **User Matrix (U)** → captures each user’s *tastes*  
- **Movie Matrix (V)** → captures each movie’s *characteristics*

When multiplied, they reconstruct the original matrix — and predict missing ratings.

```

Predicted Rating = U × Vᵀ

```

---

### Step 3: The Math Made Simple
Each user and each movie is represented as a **vector** of hidden traits (like *action*, *comedy*, *drama*).

**Example:**

| Entity        | Action | Comedy |
|----------------|---------|--------|
| User #1        | 0.9     | 0.1    |
| *Dragonheart*  | 0.9     | 0.2    |

To predict how much User #1 will like *Dragonheart*:

```

Predicted Score = (0.9 × 0.9) + (0.1 × 0.2) = 0.83
Predicted Rating ≈ 0.83 × 5 = 4.15 stars

```

A higher score → stronger recommendation.

---

### Step 4: Why It’s Called “Alternating Least Squares”

- **Alternating:**  
  The algorithm **takes turns** updating users and movies:
  1. Fix movie data → learn better user preferences.  
  2. Fix user data → learn better movie profiles.  
  3. Repeat until predictions are stable.

- **Least Squares:**  
  After each round, the algorithm reduces the **sum of squared prediction errors**, ensuring predictions get closer to real ratings.

In math terms:
```

Error = Σ (Predicted - Actual)²

````

---

## Example Walkthrough

1. Start with random user and movie vectors.
2. Predict all known ratings.
3. Compute error between predicted vs. actual.
4. Adjust user and movie vectors to reduce error.
5. Repeat (alternate) until the total error stops improving.

By the end, the system learns each user’s taste and each movie’s style.

---

## Results

### Top Recommendations for a Sample User
| Rank | Movie            | Predicted Score |
|------|------------------|----------------:|
| 1️⃣  | Dragonheart       | 1.017 |
| 2️⃣  | Primal Fear       | 0.953 |
| 3️⃣  | Amadeus           | 0.938 |

### Movies Similar to *Se7en*
| Rank | Movie                  | Similarity |
|------|-------------------------|------------|
| 1️⃣  | The Shawshank Redemption | 0.290 |
| 2️⃣  | Tombstone                | 0.286 |
| 3️⃣  | Dark City                | 0.286 |

---

## Key Terms

| Term | Meaning |
|------|----------|
| **Collaborative Filtering** | Recommending based on what similar users liked |
| **Matrix Factorization** | Splitting the user–movie table into two smaller ones |
| **Latent Features** | Hidden traits that define tastes and movie styles |
| **Sparsity** | When most ratings are missing |
| **Least Squares** | Minimizing squared prediction errors |
| **Regularization** | Prevents overfitting by penalizing large weights |

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/ALS-Recommendation-System.git
   cd ALS-Recommendation-System
   ````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**

   ```bash
   jupyter notebook ALS_Recommendation_System.ipynb
   ```

4. **Explore predictions**

   * View top recommendations for any user
   * Find similar movies using vector similarity

---

## Real-World Applications

| Platform     | Use Case                |
| ------------ | ----------------------- |
| 🎥 Netflix   | Movie recommendations   |
| 🎵 Spotify   | Song recommendations    |
| 🛍️ Amazon    | Product recommendations |
| 📺 YouTube   | Video recommendations   |
| 📷 Pinterest | Image/pin suggestions   |

---

## References

* [MovieLens Dataset (GroupLens)](https://grouplens.org/datasets/movielens/)
* [Spark MLlib ALS Documentation](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
* [Implicit Library on GitHub](https://github.com/benfred/implicit)

---

## Summary

ALS learns *hidden patterns* from millions of ratings to make personalized recommendations — turning raw data into smart insights.

> It’s like having a best friend who has seen every movie and knows exactly what you’ll love next. 

---

## Author

**[Mudassar Hakim]**
Machine Learning & Data Enthusiast
[Profile](https://www.mudassar-hakim.lovable.app)
[LinkedIn](https://www.linkedin.com/in/mudassar-ahamer-hakim-281b8b9/)

---

