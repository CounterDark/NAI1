import argparse
import os

from modules.animal_classifier import AnimalClassifier
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
    # Define path to dataset and model
    # Using absolute path logic relative to this script ensuring portability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    data_path = os.path.join(project_root, "data", "mushrooms", "agaricus-lepiota.txt")
    model_path = os.path.join(project_root, "models", "mushroom_model.keras")

    # Initialize classifier
    classifier = MushroomClassifier()

    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Load and preprocess data
    # This step converts categorical text data into numerical format suitable for neural networks.
    X_train, X_test, y_train, y_test = classifier.load_and_preprocess_data(data_path)

    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        # If a trained model exists, we load it to save time and resources.
        classifier.load_model(model_path)
    else:
        print("No existing model found. Training new model...")
        # Build model architecture (Multilayer Perceptron)
        input_dim = X_train.shape[1]
        classifier.build_model(input_dim)

        # Train the model on the training data
        classifier.train(X_train, y_train, epochs=20, batch_size=32)

        # Save the trained model for future use
        classifier.save_model(model_path)

    # Evaluate the model's performance on the test set (data it hasn't seen during training)
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

    # Define model path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    model_path = os.path.join(project_root, "models", "animal_model.keras")

    # Initialize classifier
    classifier = AnimalClassifier()

    # Load and preprocess data
    # CIFAR-10 is a standard dataset included in TensorFlow/Keras, so we don't need a local text file path.
    # Data is downloaded automatically if not present in cache.
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

        # Train the model
        # Epochs: number of times the model sees the entire dataset.
        classifier.train(train_images, train_labels, epochs=10)

        # Save the trained model
        classifier.save_model(model_path)

    # Evaluate the model
    classifier.evaluate(test_images, test_labels)


def main() -> None:
    # Create argument parser to handle command line arguments
    parser = argparse.ArgumentParser(description="Run Mushroom or Animal Classifier.")

    # Add an argument for the task type
    # 'choices' restricts the input to specific values.
    # 'required=True' ensures the user must provide this argument.
    parser.add_argument(
        "--task",
        type=str,
        choices=["mushrooms", "animals"],
        required=True,
        help="Select which classifier to run: 'mushrooms' for edible/poisonous classification, 'animals' for CIFAR-10 image classification.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Route to the appropriate function based on user input
    if args.task == "mushrooms":
        run_mushroom_classifier()
    elif args.task == "animals":
        run_animal_classifier()


if __name__ == "__main__":
    main()
