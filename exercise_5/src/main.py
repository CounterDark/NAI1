import argparse
import os

from modules.animal_classifier import AnimalClassifier
from modules.clothing_classifier import ClothingClassifier
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

    # Define model path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    model_base_path = os.path.join(project_root, "models", "clothing_model.keras")

    classifier = ClothingClassifier()

    # Load data
    train_images, test_images, train_labels, test_labels = (
        classifier.load_and_preprocess_data()
    )

    # Build models
    classifier.build_mlp_model()
    classifier.build_cnn_model()

    # Train models
    print("\n--- Training MLP Model ---")
    classifier.train_mlp(train_images, train_labels, epochs=15)

    print("\n--- Training CNN Model ---")
    classifier.train_cnn(train_images, train_labels, epochs=15)

    # Evaluate comparison
    classifier.evaluate_models(test_images, test_labels)

    # Save models
    classifier.save_models(model_base_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Mushroom, Animal, or Clothing Classifier."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["mushrooms", "animals", "clothes"],
        required=True,
        help="Select classifier: 'mushrooms', 'animals', or 'clothes'.",
    )

    args = parser.parse_args()

    if args.task == "mushrooms":
        run_mushroom_classifier()
    elif args.task == "animals":
        run_animal_classifier()
    elif args.task == "clothes":
        run_clothing_classifier()


if __name__ == "__main__":
    main()
