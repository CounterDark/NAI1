"""
Project: Multtiple model classifiers
Authors: Mateusz Anikiej and Aleksander Kunkowski

Description:
This script trains multiple model classifiers on different datasets.

Usage:
    python src/main.py --task <task>

Tasks:
    mushrooms: Train a Mushroom Classifier
    animals: Train an Animal Classifier
    clothes: Train a Clothing Classifier
    imdb: Train an IMDB Review Sentiment Classifier

Mushroom Classifier:
    - Dataset: Mushrooms
    - Model: MLP
    - Task: Classification
    - Metrics: Accuracy
Compared with the model from exercise 4.

Animal Classifier:
    - Dataset: CIFAR-10
    - Model: CNN
    - Task: Classification
    - Metrics: Accuracy
Prepared confusion matrix.

Clothing Classifier:
    - Dataset: Fashion-MNIST
    - Model: MLP, CNN
    - Task: Classification
    - Metrics: Accuracy
Created two models and compared them.

IMDB Review Sentiment Classifier:
    - Dataset: IMDB Movie Reviews
    - Model: RNN (LSTM)
    - Task: Sentiment Analysis
    - Metrics: Accuracy
"""

import argparse
import os

from modules.animal_classifier import AnimalClassifier
from modules.clothing_classifier import ClothingClassifier
from modules.imdb_review_sentiment_classifier import IMDBReviewSentimentClassifier
from modules.mushroom_classifier import MushroomClassifier


def run_mushroom_classifier() -> None:
    """
    Runs the Mushroom Classifier workflow:
    1. Loads/Preprocesses data.
    2. Builds or Loads Model.
    3. Trains (if new).
    4. Evaluates.
    """
    print("\n--- Running Mushroom Classifier ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    data_path = os.path.join(project_root, "data", "mushrooms", "agaricus-lepiota.txt")
    model_path = os.path.join(project_root, "models", "mushroom_model.keras")

    classifier = MushroomClassifier()

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Load and preprocess data
    # This step converts categorical text data into numerical format suitable for neural networks.
    X_train, X_test, y_train, y_test = classifier.load_and_preprocess_data(data_path)

    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        classifier.load_model(model_path)
    else:
        print("No existing model found. Training new model...")
        # Build model architecture (Multilayer Perceptron)
        input_dim = X_train.shape[1]
        classifier.build_model(input_dim)
        classifier.train(X_train, y_train, epochs=20, batch_size=32)
        classifier.save_model(model_path)
    classifier.evaluate(X_test, y_test)


def run_animal_classifier() -> None:
    """
    Runs the Animal Classifier (CIFAR-10) workflow:
    1. Loads/Preprocesses data.
    2. Builds or Loads Model.
    3. Trains (if new).
    4. Evaluates.
    """
    print("\n--- Running Animal Classifier (CIFAR-10) ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    model_path = os.path.join(project_root, "models", "animal_model.keras")

    classifier = AnimalClassifier()

    # Load and preprocess data
    # CIFAR-10 is a standard dataset included in TensorFlow/Keras.
    train_images, test_images, train_labels, test_labels = (
        classifier.load_and_preprocess_data()
    )

    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        classifier.load_model(model_path)
    else:
        print("No existing model found. Training new model...")
        # Build Convolutional Neural Network (CNN) architecture
        classifier.build_model()

        # Train the model using Data Augmentation to improve accuracy and generalization
        print("Training with Data Augmentation...")
        classifier.train_with_augmentation(train_images, train_labels)
        classifier.save_model(model_path)

    classifier.evaluate(test_images, test_labels)
    classifier.generate_confusion_matrix(test_images, test_labels)


def run_clothing_classifier() -> None:
    """
    Runs the Clothing Classifier (Fashion-MNIST) workflow:
    1. Loads Data.
    2. Builds both MLP and CNN models.
    3. Trains both models.
    4. Evaluates and compares them.
    """
    print("\n--- Running Clothing Classifier (Fashion-MNIST) ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    model_base_path = os.path.join(project_root, "models", "clothing_model.keras")

    classifier = ClothingClassifier()

    train_images, test_images, train_labels, test_labels = (
        classifier.load_and_preprocess_data()
    )

    classifier.build_mlp_model()
    classifier.build_cnn_model()

    print("\n--- Training MLP Model ---")
    classifier.train_mlp(train_images, train_labels, epochs=15)

    print("\n--- Training CNN Model ---")
    classifier.train_cnn(train_images, train_labels, epochs=15)
    classifier.evaluate_models(test_images, test_labels)
    classifier.save_models(model_base_path)


def run_imdb_classifier() -> None:
    """
    Runs the IMDB Review Sentiment Classifier workflow:
    1. Loads/Preprocesses data.
    2. Builds or Loads Model.
    3. Trains (if new).
    4. Evaluates.
    """
    print("\n--- Running IMDB Review Sentiment Classifier ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    model_path = os.path.join(project_root, "models", "imdb_model.keras")

    classifier = IMDBReviewSentimentClassifier()

    train_data, test_data, train_labels, test_labels = (
        classifier.load_and_preprocess_data()
    )

    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        classifier.load_model(model_path)
    else:
        print("No existing model found. Training new model...")
        classifier.build_model()
        classifier.train(train_data, train_labels, epochs=10)
        classifier.save_model(model_path)

    classifier.evaluate(test_data, test_labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Mushroom, Animal, Clothing, or IMDB Classifier."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["mushrooms", "animals", "clothes", "imdb"],
        required=True,
        help="Select classifier: 'mushrooms', 'animals', 'clothes', or 'imdb'.",
    )

    args = parser.parse_args()

    if args.task == "mushrooms":
        run_mushroom_classifier()
    elif args.task == "animals":
        run_animal_classifier()
    elif args.task == "clothes":
        run_clothing_classifier()
    elif args.task == "imdb":
        run_imdb_classifier()


if __name__ == "__main__":
    main()
