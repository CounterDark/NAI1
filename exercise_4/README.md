# Classification of Seeds and Mushrooms

This project implements classification models (Decision Tree and SVM) for two datasets: Wheat Seeds (numeric features) and Mushrooms (categorical features). It includes data preprocessing, model training, hyperparameter tuning using grid search, evaluation, and visualization of decision boundaries.

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
make run
```

Option B: using uv directly

```bash
uv run python src/main.py
```

## Project Description

The application performs the following steps for both the Seeds and Mushroom datasets:

1. **Data Loading**: Loads the dataset from text files.
2. **Preprocessing**:
   - For Seeds (numeric): Splits into train/test sets.
   - For Mushrooms (categorical): Uses Ordinal Encoding for Decision Trees and One-Hot Encoding with Scaling for SVMs.
3. **Training & Evaluation**:
   - Trains a Decision Tree classifier.
   - Performs a Grid Search to find the best hyperparameters for an SVM classifier.
   - Evaluates models using accuracy, confusion matrices, and classification reports.
4. **Model Persistence**: Saves the trained models to disk.
5. **Visualization**: Selects the top 2 features based on feature importance and visualizes the decision boundaries for both classifiers.
6. **Sample Prediction**: Demonstrates the model by predicting the class of a sample from the test set.
