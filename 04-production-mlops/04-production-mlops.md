# Phase 4: Production & MLOps (The Reality)

_Goal: Bridge the gap from model development to production-ready systems. This phase leans heavily on your Staff Engineer background._

## Phase 4 Curriculum Outline

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

---

## Unit Details

### Unit 13: MLOps and Deployment

- **Topics to Learn:**
  - **Model Versioning:** Using tools like DVC or MLflow to track experiments and model artifacts.
  - **Containerization:** Packaging your model and its dependencies into a Docker image.
  - **API Serving:** Using a web framework like FastAPI or Flask to create an API endpoint for your model.
  - **Monitoring:** Concepts of data drift and model performance degradation.
- **Curated Resources:**
  - [DataCamp - MLOps Tutorial](https://www.datacamp.com/tutorial/tutorial-machine-learning-pipelines-mlops-deployment)
  - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
  - [FastAPI Documentation](https://fastapi.tiangolo.com/)
- **Challenge Clarified:** Deploy a model as a REST API.
  - **Concrete Example:** Take the sentiment classifier model you fine-tuned in Unit 11. Wrap it in a FastAPI application with a /predict endpoint that accepts a JSON payload with text and returns the sentiment. Package this entire application with a Dockerfile. The final deliverable is a Docker image that can be run anywhere to serve your model.

### Unit 14: Ethical & Responsible AI

- **Topics to Learn:**
  - **Sources of Bias:** Data bias, algorithmic bias, and human bias.
  - **Fairness Metrics:** Understanding metrics like demographic parity and equal opportunity.
  - **Explainability (XAI):** Using tools like SHAP or LIME to understand _why_ a model made a specific prediction.
- **Curated Resources:**
  - [Google - Introduction to Responsible AI (Course)](https://www.cloudskillsboost.google/course_templates/536)
  - [CITI Program - Essentials of Responsible AI (Course)](https://about.citiprogram.org/course/essentials-of-responsible-ai/)
- **Challenge Clarified:** Perform a model bias audit.
  - **Concrete Example:** Use a dataset like the "Adult Census Income" dataset, which is often used to predict if income exceeds $50K/yr. Train a classifier. Then, use tools to analyze if the model's prediction error rate is significantly different for different demographic groups (e.g., based on gender or race). Document your findings and suggest a mitigation strategy (e.g., re-sampling the data).

### Unit 15: Advanced Systems & Optimization

- **Topics to Learn:**
  - **Optimizers:** Adam, RMSprop, and how they differ from standard gradient descent.
  - **Distributed Training Concepts:** Data Parallelism vs. Model Parallelism.
  - **High-Level Understanding:** What problems systems like Horovod (for data parallelism) and DeepSpeed (for large model training) are designed to solve.
- **Curated Resources:**
  - [Coursera - Advanced Learning Algorithms](https://www.coursera.org/learn/advanced-learning-algorithms)
  - [UW-Madison CS 839 - Advanced ML Systems (Academic Course)](https://pages.cs.wisc.edu/~shivaram/cs839-sp22/)
- **Challenge Clarified:** Research and summarize a distributed training system.
  - **Concrete Example:** Read the introductory blog posts or documentation for Horovod. Write a one-page technical summary explaining: 1) What problem it solves (syncing gradients across multiple GPUs). 2) How its ring-allreduce algorithm works at a high level. 3) What code changes are needed to adapt a standard PyTorch training script to use Horovod.
