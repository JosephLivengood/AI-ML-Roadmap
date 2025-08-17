# Phase 3: Deep Learning & Modern AI (The Frontier)

_Goal: Move from traditional ML to modern deep learning. This is where you'll build the skills for today's most advanced AI applications._

## Phase 3 Curriculum Outline

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

---

## Unit Details

### Unit 7: Neural Networks and Backpropagation

- **Topics to Learn:**
  - **Architecture:** Perceptrons, Layers (Input, Hidden, Output), and Activation Functions (Sigmoid, Tanh, ReLU).
  - **Forward Propagation:** The process of passing inputs through the network to get an output.
  - **Backpropagation:** The process of using the chain rule to calculate the gradient of the loss function with respect to each weight in the network.
- **Curated Resources:**
  - [Neural Networks and Deep Learning (Online Book)](http://neuralnetworksanddeeplearning.com/)
  - [Neural Networks from Scratch in Python (Book/Course)](https://nnfs.io/)
- **Challenge Clarified:** Build a neural network from scratch for MNIST.
  - **Concrete Example:** Using only Python and NumPy, create a class for your neural network. It should have methods for forward(), backward(), and update_weights(). Train this network on the MNIST dataset of handwritten digits. The goal is not to achieve state-of-the-art accuracy, but to prove you can implement the mechanics of forward and backward passes correctly.

### Unit 8: Convolutional Neural Networks (CV)

- **Topics to Learn:**
  - **Convolutional Layers:** The concepts of kernels (filters), stride, and padding.
  - **Pooling Layers:** Max Pooling and Average Pooling for down-sampling.
  - **CNN Architectures:** Understand the structure of stacking convolutional and pooling layers, followed by dense layers for classification.
  - **PyTorch:** nn.Conv2d, nn.MaxPool2d, nn.Linear, and the training loop structure.
- **Curated Resources:**
  - [PyTorch - Training a Classifier (Official Tutorial)](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
  - [Stanford CS231n - CNNs for Visual Recognition (Course)](http://cs231n.stanford.edu/)
- **Challenge Clarified:** Build and train a custom CNN for image classification.
  - **Concrete Example:** Using PyTorch, define a simple CNN architecture (e.g., two convolutional layers, each followed by a pooling layer, then a final dense layer). Train this network on the CIFAR-10 dataset. The goal is to get comfortable with the PyTorch workflow for defining, training, and evaluating a model on a standard image dataset.

### Unit 9: Recurrent Neural Networks (NLP/Sequences)

- **Topics to Learn:**
  - **Handling Sequences:** The concept of a hidden state that carries information from one timestep to the next.
  - **RNN Issues:** Vanishing and exploding gradient problems.
  - **LSTM & GRU:** The high-level architecture of these cells, including their gates (e.g., forget, input, output gates in LSTM), which solve the vanishing gradient problem.
- **Curated Resources:**
  - [Understanding LSTMs - Chris Olah (Article)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - [Stanford CS224n - NLP with Deep Learning (Course)](https://web.stanford.edu/class/cs224n/)
- **Challenge Clarified:** Use an LSTM for time-series prediction or text generation.
  - **Concrete Example (Time Series):** Take a dataset of daily stock prices. Train an LSTM to predict the next day's closing price based on the previous 30 days.
  - **Concrete Example (Text Gen):** Train an LSTM on a body of text (e.g., Shakespeare). After training, provide it with a starting phrase ("To be or") and have it generate the next sequence of characters or words.

### Unit 10: Attention and Transformers

- **Topics to Learn:**
  - **Attention Intuition:** The idea of allowing the model to weigh the importance of different parts of the input sequence.
  - **Transformer Architecture:** Self-Attention, Multi-Head Attention, Positional Encodings, and the Encoder-Decoder structure.
- **Curated Resources:**
  - [The Illustrated Transformer - Jay Alammar (Article)](https://jalammar.github.io/illustrated-transformer/)
- **Challenge Clarified:** Implement a simplified transformer model.
  - **Concrete Example:** Focus on implementing just the self-attention mechanism from scratch. Write a function that takes in a set of input vectors and computes the Query, Key, and Value matrices, performs the scaled dot-product attention, and returns the weighted output vectors. This demonstrates you understand the core component of the transformer.

### Unit 11: Large Language Models (LLMs)

- **Topics to Learn:**
  - **Pre-training vs. Fine-tuning:** The two-stage process that makes LLMs powerful.
  - **Fine-tuning Strategies:** Full fine-tuning vs. parameter-efficient methods (like LoRA).
  - **Prompt Engineering:** Techniques for crafting effective prompts to guide model behavior.
  - **Hugging Face:** Using the Trainer API and the transformers library to load, fine-tune, and use pre-trained models.
- **Curated Resources:**
  - [Hugging Face NLP Course](https://huggingface.co/course)
- **Challenge Clarified:** Fine-tune a pre-trained model for a domain task.
  - **Concrete Example:** Use the Hugging Face library to load a pre-trained model like distilbert-base-uncased. Find a dataset for a specific text classification task (e.g., classifying customer reviews as positive/negative). Fine-tune the model on this dataset and demonstrate that its performance on your specific task is better than the base model's.

### Unit 12: Generative AI

- **Topics to Learn:**
  - **Variational Autoencoders (VAEs):** The concept of encoding an input into a latent space (a probability distribution) and decoding from it.
  - **Generative Adversarial Networks (GANs):** The two-player game between a Generator and a Discriminator.
  - **Diffusion Models:** The high-level concept of adding noise to an image and then training a model to reverse the process.
- **Curated Resources:**
  - [GeeksforGeeks - Generative Models: GANs and VAEs (Article)](https://www.geeksforgeeks.org/deep-learning/generative-models-in-ai-a-comprehensive-comparison-of-gans-and-vaes/)
  - [Lil'Log - Lillian Weng's Blog (Advanced Topics)](https://lilianweng.github.io/)
- **Challenge Clarified:** Train a simple GAN or VAE.
  - **Concrete Example:** Using PyTorch or TensorFlow, build a simple GAN to generate MNIST digits. Your Generator network will take random noise and try to create a 28x28 image. Your Discriminator will take an image (real or fake) and classify it. The goal is to see the generated images become more digit-like over many training epochs.
