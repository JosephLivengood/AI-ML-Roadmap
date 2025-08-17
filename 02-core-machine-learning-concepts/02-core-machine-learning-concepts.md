# Phase 2: Core Machine Learning Concepts (The Toolkit)

_Goal: Understand the "what" and "why" behind traditional ML models. Implement them from scratch before relying on frameworks to build deep intuition._

## Phase 2 Curriculum Outline

- **Unit 4: Supervised Learning Algorithms**
  - [ ] Linear & Logistic Regression
  - [ ] Decision Trees
  - [ ] Support Vector Machines (SVMs)
  - [ ] **Challenge:** Build a simple decision tree from scratch.
- **Unit 5: Unsupervised Learning**
  - [ ] Clustering
  - [ ] Dimensionality Reduction
  - [ ] Feature Selection vs. Extraction
  - [ ] **Challenge:** Customer segmentation analysis.
- **Unit 6: Model Building & Evaluation**
  - [ ] Bias-Variance Tradeoff
  - [ ] Cross-Validation
  - [ ] Metrics Deep Dive
  - [ ] **Challenge:** Create a comprehensive model evaluation pipeline.

---

## Unit Details

### Unit 4: Supervised Learning Algorithms

- **Topics to Learn:**
  - **Linear & Logistic Regression:** The underlying math, cost functions (MSE, Log Loss), and optimization with gradient descent.
  - **Decision Trees:** Concepts of entropy, information gain, and how the tree is recursively split.
  - **Support Vector Machines (SVMs):** The intuition of finding the maximum margin hyperplane.
- **Curated Resources:**
  - [An Introduction to Statistical Learning (Textbook)](https://www.statlearning.com/)
  - [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- **Challenge Clarified:** Build a simple decision tree from scratch.
  - **Concrete Example:** Using a small, clean dataset (e.g., a subset of the Iris dataset), write Python code to find the best feature and split-point that maximizes information gain. Recursively apply this logic to build a tree. Then, train a sklearn.tree.DecisionTreeClassifier on the same data and compare the structure of your tree and its accuracy to the library's version.

### Unit 5: Unsupervised Learning

- **Topics to Learn:**
  - **Clustering:** K-Means algorithm (how centroids are updated), hierarchical clustering.
  - **Dimensionality Reduction:** The practical application of PCA and the intuition behind t-SNE for visualization.
  - **Feature Selection vs. Extraction:** The difference between choosing existing features and creating new ones.
- **Curated Resources:**
  - [GeeksforGeeks - Unsupervised Learning Examples (Article)](https://www.geeksforgeeks.org/machine-learning/unsupervised-machine-learning-examples/)
  - [Scikit-Learn Guide on Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- **Challenge Clarified:** Customer segmentation analysis.
  - **Concrete Example:** Use a dataset like the "Mall Customer Segmentation Data" on Kaggle. Apply K-Means clustering to group customers based on features like age, annual income, and spending score. Then, use PCA or t-SNE to visualize these clusters in 2D. The final notebook should present these visualizations and provide business-oriented interpretations of each customer segment (e.g., "Cluster 1: Young High-Spenders").

### Unit 6: Model Building & Evaluation

- **Topics to Learn:**
  - **Bias-Variance Tradeoff:** The core concept of underfitting vs. overfitting.
  - **Cross-Validation:** K-Fold and Stratified K-Fold techniques.
  - **Metrics Deep Dive:** When to use Accuracy vs. Precision, Recall, F1-score (for imbalanced classes). How to interpret a ROC curve and AUC.
- **Curated Resources:**
  - [Scikit-Learn Guide on Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **Challenge Clarified:** Create a comprehensive model evaluation pipeline.
  - **Concrete Example:** Write a Python class or a set of functions that takes a model, training data, and test data as input. The pipeline should automatically perform K-Fold cross-validation, train the model, make predictions, and output a dictionary containing multiple key metrics (e.g., accuracy, precision, recall, F1, and AUC). This becomes a reusable tool for all future projects.
