# ML/AI Learning Path

This document outlines a self-study curriculum designed for an experienced software engineer to transition into a senior-level Machine Learning or AI role.

### Phase 1: Foundational Knowledge (The Bedrock)

_Goal: Solidify the mathematical and programming fundamentals upon which all of ML is built. Your existing coding skills are a major advantage here._

- **Unit 1: Linear Algebra Foundations**
  - **Topics to Learn:**
    - **Vectors:** Representation, norm (magnitude), dot product, and their geometric interpretations.
    - **Matrices:** Representation, transpose, matrix multiplication, and identity matrix.
    - **Linear Transformations:** How matrices transform vectors (scaling, rotation, shearing).
    - **Eigenvectors & Eigenvalues:** Understanding them as the vectors that are only scaled by a transformation.
    - **NumPy Implementation:** np.dot(), np.linalg.eig(), @ operator for matrix multiplication, slicing, and array manipulation.
  - **Challenge Clarified:** Implement PCA from scratch.
    - **Concrete Example:** Take a simple dataset like Iris. Use NumPy to first calculate the covariance matrix of the data. Then, compute the eigenvectors and eigenvalues of this covariance matrix. Finally, project the original data onto the principal components (the top eigenvectors) to reduce its dimensionality. The goal is to show you understand the step-by-step mechanics, not just how to call sklearn.decomposition.PCA.
- **Unit 2: Statistics and Probability**
  - **Topics to Learn:**
    - **Descriptive Statistics:** Mean, median, mode, variance, standard deviation.
    - **Probability Distributions:** Uniform, Normal (Gaussian), and Binomial distributions. Understand PDFs and CDFs.
    - **Core Concepts:** Conditional probability, Bayes' Theorem.
    - **Hypothesis Testing:** The concepts of null hypothesis, p-value, and t-tests.
  - **Challenge Clarified:** Perform and document a statistical analysis.
    - **Concrete Example:** Use the Titanic dataset from Kaggle. In a Jupyter Notebook, formulate a hypothesis, such as "Passengers who paid a higher fare had a higher survival rate." Use Pandas for data cleaning, Matplotlib/Seaborn for an Exploratory Data Analysis (EDA) to visualize relationships, and a statistical test (like an independent t-test) to formally test your hypothesis. The final document should clearly state your question, show your visual and statistical analysis, and conclude with whether you can reject the null hypothesis.
- **Unit 3: Calculus for Machine Learning**
  - **Topics to Learn:**
    - **Derivatives:** Understanding the derivative as a rate of change or slope.
    - **The Chain Rule:** The fundamental concept that allows for backpropagation.
    - **Partial Derivatives & Gradients:** How to find the "slope" in multiple dimensions. The gradient is a vector of partial derivatives that points in the direction of steepest ascent.
    - **Optimization:** The concept of finding the minimum of a function by moving in the opposite direction of the gradient.
  - **Challenge Clarified:** Implement gradient descent variants.
    - **Concrete Example:** Create a simple linear regression problem with one variable (y = mx + b). Write three Python functions: one for standard "batch" gradient descent (calculates gradient on all data), one for stochastic (calculates on one data point at a time), and one for mini-batch. Plot the "loss" (e.g., Mean Squared Error) over iterations for each variant to see how their convergence behavior differs.

### Phase 2: Core Machine Learning Concepts (The Toolkit)

_Goal: Understand the "what" and "why" behind traditional ML models. Implement them from scratch before relying on frameworks to build deep intuition._

- **Unit 4: Supervised Learning Algorithms**
  - **Topics to Learn:**
    - **Linear & Logistic Regression:** The underlying math, cost functions (MSE, Log Loss), and optimization with gradient descent.
    - **Decision Trees:** Concepts of entropy, information gain, and how the tree is recursively split.
    - **Support Vector Machines (SVMs):** The intuition of finding the maximum margin hyperplane.
  - **Challenge Clarified:** Build a simple decision tree from scratch.
    - **Concrete Example:** Using a small, clean dataset (e.g., a subset of the Iris dataset), write Python code to find the best feature and split-point that maximizes information gain. Recursively apply this logic to build a tree. Then, train a sklearn.tree.DecisionTreeClassifier on the same data and compare the structure of your tree and its accuracy to the library's version.
- **Unit 5: Unsupervised Learning**
  - **Topics to Learn:**
    - **Clustering:** K-Means algorithm (how centroids are updated), hierarchical clustering.
    - **Dimensionality Reduction:** The practical application of PCA and the intuition behind t-SNE for visualization.
    - **Feature Selection vs. Extraction:** The difference between choosing existing features and creating new ones.
  - **Challenge Clarified:** Customer segmentation analysis.
    - **Concrete Example:** Use a dataset like the "Mall Customer Segmentation Data" on Kaggle. Apply K-Means clustering to group customers based on features like age, annual income, and spending score. Then, use PCA or t-SNE to visualize these clusters in 2D. The final notebook should present these visualizations and provide business-oriented interpretations of each customer segment (e.g., "Cluster 1: Young High-Spenders").
- **Unit 6: Model Building & Evaluation**
  - **Topics to Learn:**
    - **Bias-Variance Tradeoff:** The core concept of underfitting vs. overfitting.
    - **Cross-Validation:** K-Fold and Stratified K-Fold techniques.
    - **Metrics Deep Dive:** When to use Accuracy vs. Precision, Recall, F1-score (for imbalanced classes). How to interpret a ROC curve and AUC.
  - **Challenge Clarified:** Create a comprehensive model evaluation pipeline.
    - **Concrete Example:** Write a Python class or a set of functions that takes a model, training data, and test data as input. The pipeline should automatically perform K-Fold cross-validation, train the model, make predictions, and output a dictionary containing multiple key metrics (e.g., accuracy, precision, recall, F1, and AUC). This becomes a reusable tool for all future projects.

### Phase 3: Deep Learning & Modern AI (The Frontier)

_Goal: Move from traditional ML to modern deep learning. This is where you'll build the skills for today's most advanced AI applications._

- **Unit 7: Neural Networks and Backpropagation**
  - **Topics to Learn:**
    - **Architecture:** Perceptrons, Layers (Input, Hidden, Output), and Activation Functions (Sigmoid, Tanh, ReLU).
    - **Forward Propagation:** The process of passing inputs through the network to get an output.
    - **Backpropagation:** The process of using the chain rule to calculate the gradient of the loss function with respect to each weight in the network.
  - **Challenge Clarified:** Build a neural network from scratch for MNIST.
    - **Concrete Example:** Using only Python and NumPy, create a class for your neural network. It should have methods for forward(), backward(), and update_weights(). Train this network on the MNIST dataset of handwritten digits. The goal is not to achieve state-of-the-art accuracy, but to prove you can implement the mechanics of forward and backward passes correctly.
- **Unit 8: Convolutional Neural Networks (CV)**
  - **Topics to Learn:**
    - **Convolutional Layers:** The concepts of kernels (filters), stride, and padding.
    - **Pooling Layers:** Max Pooling and Average Pooling for down-sampling.
    - **CNN Architectures:** Understand the structure of stacking convolutional and pooling layers, followed by dense layers for classification.
    - **PyTorch:** nn.Conv2d, nn.MaxPool2d, nn.Linear, and the training loop structure.
  - **Challenge Clarified:** Build and train a custom CNN for image classification.
    - **Concrete Example:** Using PyTorch, define a simple CNN architecture (e.g., two convolutional layers, each followed by a pooling layer, then a final dense layer). Train this network on the CIFAR-10 dataset. The goal is to get comfortable with the PyTorch workflow for defining, training, and evaluating a model on a standard image dataset.
- **Unit 9: Recurrent Neural Networks (NLP/Sequences)**
  - **Topics to Learn:**
    - **Handling Sequences:** The concept of a hidden state that carries information from one timestep to the next.
    - **RNN Issues:** Vanishing and exploding gradient problems.
    - **LSTM & GRU:** The high-level architecture of these cells, including their gates (e.g., forget, input, output gates in LSTM), which solve the vanishing gradient problem.
  - **Challenge Clarified:** Use an LSTM for time-series prediction or text generation.
    - **Concrete Example (Time Series):** Take a dataset of daily stock prices. Train an LSTM to predict the next day's closing price based on the previous 30 days.
    - **Concrete Example (Text Gen):** Train an LSTM on a body of text (e.g., Shakespeare). After training, provide it with a starting phrase ("To be or") and have it generate the next sequence of characters or words.
- **Unit 10: Attention and Transformers**
  - **Topics to Learn:**
    - **Attention Intuition:** The idea of allowing the model to weigh the importance of different parts of the input sequence.
    - **Transformer Architecture:** Self-Attention, Multi-Head Attention, Positional Encodings, and the Encoder-Decoder structure.
  - **Challenge Clarified:** Implement a simplified transformer model.
    - **Concrete Example:** Focus on implementing just the self-attention mechanism from scratch. Write a function that takes in a set of input vectors and computes the Query, Key, and Value matrices, performs the scaled dot-product attention, and returns the weighted output vectors. This demonstrates you understand the core component of the transformer.
- **Unit 11: Large Language Models (LLMs)**
  - **Topics to Learn:**
    - **Pre-training vs. Fine-tuning:** The two-stage process that makes LLMs powerful.
    - **Fine-tuning Strategies:** Full fine-tuning vs. parameter-efficient methods (like LoRA).
    - **Prompt Engineering:** Techniques for crafting effective prompts to guide model behavior.
    - **Hugging Face:** Using the Trainer API and the transformers library to load, fine-tune, and use pre-trained models.
  - **Challenge Clarified:** Fine-tune a pre-trained model for a domain task.
    - **Concrete Example:** Use the Hugging Face library to load a pre-trained model like distilbert-base-uncased. Find a dataset for a specific text classification task (e.g., classifying customer reviews as positive/negative). Fine-tune the model on this dataset and demonstrate that its performance on your specific task is better than the base model's.
- **Unit 12: Generative AI**
  - **Topics to Learn:**
    - **Variational Autoencoders (VAEs):** The concept of encoding an input into a latent space (a probability distribution) and decoding from it.
    - **Generative Adversarial Networks (GANs):** The two-player game between a Generator and a Discriminator.
    - **Diffusion Models:** The high-level concept of adding noise to an image and then training a model to reverse the process.
  - **Challenge Clarified:** Train a simple GAN or VAE.
    - **Concrete Example:** Using PyTorch or TensorFlow, build a simple GAN to generate MNIST digits. Your Generator network will take random noise and try to create a 28x28 image. Your Discriminator will take an image (real or fake) and classify it. The goal is to see the generated images become more digit-like over many training epochs.

### Phase 4: Production & MLOps (The Reality)

_Goal: Bridge the gap from model development to production-ready systems. This phase leans heavily on your Staff Engineer background._

- **Unit 13: MLOps and Deployment**
  - **Topics to Learn:**
    - **Model Versioning:** Using tools like DVC or MLflow to track experiments and model artifacts.
    - **Containerization:** Packaging your model and its dependencies into a Docker image.
    - **API Serving:** Using a web framework like FastAPI or Flask to create an API endpoint for your model.
    - **Monitoring:** Concepts of data drift and model performance degradation.
  - **Challenge Clarified:** Deploy a model as a REST API.
    - **Concrete Example:** Take the sentiment classifier model you fine-tuned in Unit 11. Wrap it in a FastAPI application with a /predict endpoint that accepts a JSON payload with text and returns the sentiment. Package this entire application with a Dockerfile. The final deliverable is a Docker image that can be run anywhere to serve your model.
- **Unit 14: Ethical & Responsible AI**
  - **Topics to Learn:**
    - **Sources of Bias:** Data bias, algorithmic bias, and human bias.
    - **Fairness Metrics:** Understanding metrics like demographic parity and equal opportunity.
    - **Explainability (XAI):** Using tools like SHAP or LIME to understand _why_ a model made a specific prediction.
  - **Challenge Clarified:** Perform a model bias audit.
    - **Concrete Example:** Use a dataset like the "Adult Census Income" dataset, which is often used to predict if income exceeds $50K/yr. Train a classifier. Then, use tools to analyze if the model's prediction error rate is significantly different for different demographic groups (e.g., based on gender or race). Document your findings and suggest a mitigation strategy (e.g., re-sampling the data).
- **Unit 15: Advanced Systems & Optimization**
  - **Topics to Learn:**
    - **Optimizers:** Adam, RMSprop, and how they differ from standard gradient descent.
    - **Distributed Training Concepts:** Data Parallelism vs. Model Parallelism.
    - **High-Level Understanding:** What problems systems like Horovod (for data parallelism) and DeepSpeed (for large model training) are designed to solve.
  - **Challenge Clarified:** Research and summarize a distributed training system.
    - **Concrete Example:** Read the introductory blog posts or documentation for Horovod. Write a one-page technical summary explaining: 1) What problem it solves (syncing gradients across multiple GPUs). 2) How its ring-allreduce algorithm works at a high level. 3) What code changes are needed to adapt a standard PyTorch training script to use Horovod.

### Phase 5: Building a Senior-Level Portfolio

_Goal: Synthesize all learned skills into tangible proof of your capabilities. The challenges from the previous units form the core of this portfolio._

- **Capstone Project:** Select one of the more complex challenge projects (e.g., the fine-tuned LLM or the deployed API) and polish it into a portfolio centerpiece. Write a detailed blog post or README explaining the problem, your process, the architecture, and the results.
- **ML System Design:** Practice ML system design interview questions. Whiteboard the architecture for systems like a real-time fraud detection service or a personalized news feed. This demonstrates senior-level thinking.
- **Paper Implementation:** Pick a foundational paper in your area of interest (e.g., "Attention Is All You Need") and implement it. This is a significant differentiator that proves deep understanding.

### Curriculum Outline

- **Phase 1: Foundational Knowledge (The Bedrock)**
  - **Unit 1: Linear Algebra Foundations**
    - [ ] Vectors
    - [ ] Matrices
    - [ ] Linear Transformations
    - [ ] Eigenvectors & Eigenvalues
    - [ ] NumPy Implementation
    - [ ] **Challenge:** Implement PCA from scratch.
  - **Unit 2: Statistics and Probability**
    - [ ] Descriptive Statistics
    - [ ] Probability Distributions
    - [ ] Core Concepts
    - [ ] Hypothesis Testing
    - [ ] **Challenge:** Perform and document a statistical analysis.
  - **Unit 3: Calculus for Machine Learning**
    - [ ] Derivatives
    - [ ] The Chain Rule
    - [ ] Partial Derivatives & Gradients
    - [ ] Optimization
    - [ ] **Challenge:** Implement gradient descent variants.
- **Phase 2: Core Machine Learning Concepts (The Toolkit)**
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
- **Phase 3: Deep Learning & Modern AI (The Frontier)**
  - **Unit 7: Neural Networks and Backpropagation**
    - [ ] Architecture
    - [ ] Forward Propagation
    - [ ] Backpropagation
    - [ ] **Challenge:** Build a neural network from scratch for MNIST.
  - **Unit 8: Convolutional Neural Networks (CV)**
    - [ ] Convolutional Layers
    - [ ] Pooling Layers
    - [ ] CNN Architectures
    - [ ] PyTorch
    - [ ] **Challenge:** Build and train a custom CNN for image classification.
  - **Unit 9: Recurrent Neural Networks (NLP/Sequences)**
    - [ ] Handling Sequences
    - [ ] RNN Issues
    - [ ] LSTM & GRU
    - [ ] **Challenge:** Use an LSTM for time-series prediction or text generation.
  - **Unit 10: Attention and Transformers**
    - [ ] Attention Intuition
    - [ ] Transformer Architecture
    - [ ] **Challenge:** Implement a simplified transformer model.
  - **Unit 11: Large Language Models (LLMs)**
    - [ ] Pre-training vs. Fine-tuning
    - [ ] Fine-tuning Strategies
    - [ ] Prompt Engineering
    - [ ] Hugging Face
    - [ ] **Challenge:** Fine-tune a pre-trained model for a domain task.
  - **Unit 12: Generative AI**
    - [ ] Variational Autoencoders (VAEs)
    - [ ] Generative Adversarial Networks (GANs)
    - [ ] Diffusion Models
    - [ ] **Challenge:** Train a simple GAN or VAE.
- **Phase 4: Production & MLOps (The Reality)**
  - **Unit 13: MLOps and Deployment**
    - [ ] Model Versioning
    - [ ] Containerization
    - [ ] API Serving
    - [ ] Monitoring
    - [ ] **Challenge:** Deploy a model as a REST API.
  - **Unit 14: Ethical & Responsible AI**
    - [ ] Sources of Bias
    - [ ] Fairness Metrics
    - [ ] Explainability (XAI)
    - [ ] **Challenge:** Perform a model bias audit.
  - **Unit 15: Advanced Systems & Optimization**
    - [ ] Optimizers
    - [ ] Distributed Training Concepts
    - [ ] High-Level Understanding
    - [ ] **Challenge:** Research and summarize a distributed training system.
- **Phase 5: Building a Senior-Level Portfolio**
  - [ ] **Capstone Project**
  - [ ] **ML System Design**
  - [ ] **Paper Implementation**
