# Created multiple models with different datasets

## Authors

- Mateusz Anikiej
- Aleksander Kunkowski

## Prerequisites

1. Python 3.13 installed
2. `uv` installed (https://docs.astral.sh/uv/getting-started/installation/)

## Setup

Use `uv` to create a virtual environment and install dependencies.

Option A: using Makefile

```bash
make setup
```

Option B: using uv directly

```bash
uv sync
```

## Running the program

You can run the classification script using the Makefile or directly with uv.

Option A: using Makefile

```bash
make exe5 ARGS="--task <task>"
```

Option B: using uv directly

```bash
uv run python src/main.py
```

## Project Description

This project, developed by Mateusz Anikiej and Aleksander Kunkowski, implements and compares various machine learning classifiers using the TensorFlow/Keras framework.

The project is structured into four main classification tasks:

1.  **Mushroom Classification**:

    - **Goal**: Classify mushrooms as edible or poisonous based on physical characteristics.
    - **Dataset**: Agaricus-Lepiota dataset (tabular data).
    - **Model**: Multi-Layer Perceptron (MLP).
    - **Key Features**: Includes data preprocessing (One-Hot Encoding) and comparison with the model from Exercise 4.

2.  **Animal Classification (CIFAR-10)**:

    - **Goal**: Classify images into 10 categories (airplane, automobile, bird, etc.) to recognise animals.
    - **Dataset**: CIFAR-10 (image data).
    - **Model**: Convolutional Neural Network (CNN).
    - **Key Features**: Utilizes data augmentation to improve generalization and includes confusion matrix generation for performance analysis.

3.  **Clothing Classification (Fashion-MNIST)**:

    - **Goal**: Classify images of clothing items into 10 categories.
    - **Dataset**: Fashion-MNIST (grayscale image data).
    - **Models**: Compares two architectures: a simple Dense Network (MLP) vs. a Convolutional Neural Network (CNN).
    - **Key Features**: Demonstrates the performance advantage of CNNs over MLPs for image data.

4.  **IMDB Review Sentiment Analysis**:
    - **Goal**: Classify movie reviews as positive or negative.
    - **Dataset**: IMDB Movie Reviews (text data).
    - **Model**: Recurrent Neural Network (RNN) with LSTM (Long Short-Term Memory) layers.
    - **Key Features**: Implements text processing (tokenization, padding) and sequence modeling for sentiment analysis.
