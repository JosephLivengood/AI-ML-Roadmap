# Phase 1: Foundational Knowledge (The Bedrock)

_Goal: Solidify the mathematical and programming fundamentals upon which all of ML is built. Your existing coding skills are a major advantage here._

## Phase 1 Curriculum Outline

- **Unit 1: Linear Algebra Foundations**
  - [x] Vectors
  - [x] Matrices
  - [x] Linear Transformations
  - [x] Eigenvectors & Eigenvalues
  - [x] NumPy Implementation
  - [x] **Challenge:** Implement PCA from scratch.
- **Unit 2: Statistics and Probability**
  - [x] Descriptive Statistics
  - [x] Probability Distributions
  - [x] Core Concepts
  - [x] Hypothesis Testing
  - [x] **Challenge:** Perform and document a statistical analysis.
- **Unit 3: Calculus for Machine Learning**
  - [x] Derivatives
  - [x] The Chain Rule
  - [x] Partial Derivatives & Gradients
  - [x] Optimization
  - [x] **Challenge:** Implement gradient descent variants.

---

## Unit Details

### Unit 1: Linear Algebra Foundations

- **Topics to Learn:**
  - **Vectors:** Representation, norm (magnitude), dot product, and their geometric interpretations.
  - **Matrices:** Representation, transpose, matrix multiplication, and identity matrix.
  - **Linear Transformations:** How matrices transform vectors (scaling, rotation, shearing).
  - **Eigenvectors & Eigenvalues:** Understanding them as the vectors that are only scaled by a transformation.
  - **NumPy Implementation:** np.dot(), np.linalg.eig(), @ operator for matrix multiplication, slicing, and array manipulation.
- **Curated Resources:**
  - [3Blue1Brown - Essence of Linear Algebra (Video Intuition)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
  - [MIT 18.06 Linear Algebra - Gilbert Strang (University Course)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
  - [Coursera - Mathematics for Machine Learning: Linear Algebra](https://www.coursera.org/learn/linear-algebra-machine-learning)
- **Challenge Clarified:** Implement PCA from scratch.
  - **Concrete Example:** Take a simple dataset like Iris. Use NumPy to first calculate the covariance matrix of the data. Then, compute the eigenvectors and eigenvalues of this covariance matrix. Finally, project the original data onto the principal components (the top eigenvectors) to reduce its dimensionality. The goal is to show you understand the step-by-step mechanics, not just how to call sklearn.decomposition.PCA.

### Unit 2: Statistics and Probability

- **Topics to Learn:**
  - **Descriptive Statistics:** Mean, median, mode, variance, standard deviation.
  - **Probability Distributions:** Uniform, Normal (Gaussian), and Binomial distributions. Understand PDFs and CDFs.
  - **Core Concepts:** Conditional probability, Bayes' Theorem.
  - **Hypothesis Testing:** The concepts of null hypothesis, p-value, and t-tests.
- **Curated Resources:**
  - [Khan Academy - Statistics and Probability (All-in-One)](https://www.khanacademy.org/math/statistics-probability)
  - [Coursera - Probability & Statistics for ML & Data Science](https://www.coursera.org/learn/probability-statistics-machine-learning-data-science)
- **Challenge Clarified:** Perform and document a statistical analysis.
  - **Concrete Example:** Use the Titanic dataset from Kaggle. In a Jupyter Notebook, formulate a hypothesis, such as "Passengers who paid a higher fare had a higher survival rate." Use Pandas for data cleaning, Matplotlib/Seaborn for an Exploratory Data Analysis (EDA) to visualize relationships, and a statistical test (like an independent t-test) to formally test your hypothesis. The final document should clearly state your question, show your visual and statistical analysis, and conclude with whether you can reject the null hypothesis.

### Unit 3: Calculus for Machine Learning

- **Topics to Learn:**
  - **Derivatives:** Understanding the derivative as a rate of change or slope.
  - **The Chain Rule:** The fundamental concept that allows for backpropagation.
  - **Partial Derivatives & Gradients:** How to find the "slope" in multiple dimensions. The gradient is a vector of partial derivatives that points in the direction of steepest ascent.
  - **Optimization:** The concept of finding the minimum of a function by moving in the opposite direction of the gradient.
- **Curated Resources:**
  - [3Blue1Brown - Essence of Calculus (Video Intuition)](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t57w)
  - [GeeksforGeeks - Calculus for Machine Learning (Article)](https://www.geeksforgeeks.org/machine-learning/mastering-calculus-for-machine-learning-key-concepts-and-applications/)
  - [YouTube - An introduction to Calculus for Machine Learning](https://www.youtube.com/watch?v=MDL384gsAk0)
- **Challenge Clarified:** Implement gradient descent variants.
  - **Concrete Example:** Create a simple linear regression problem with one variable (y = mx + b). Write three Python functions: one for standard "batch" gradient descent (calculates gradient on all data), one for stochastic (calculates on one data point at a time), and one for mini-batch. Plot the "loss" (e.g., Mean Squared Error) over iterations for each variant to see how their convergence behavior differs.
