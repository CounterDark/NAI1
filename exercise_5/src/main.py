import os

from modules.mushroom_classifier import MushroomClassifier


def main() -> None:
    # Define path to dataset and model
    # Using absolute path logic or relative to this script for safety
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    data_path = os.path.join(project_root, "data", "mushrooms", "agaricus-lepiota.txt")
    model_path = os.path.join(project_root, "models", "mushroom_model.keras")

    # Initialize classifier
    classifier = MushroomClassifier()

    # Load and preprocess data (needed for evaluation or training to get input shape)
    if os.path.exists(data_path):
        X_train, X_test, y_train, y_test = classifier.load_and_preprocess_data(
            data_path
        )

        if os.path.exists(model_path):
            print(f"Found existing model at {model_path}")
            classifier.load_model(model_path)
        else:
            print("No existing model found. Training new model...")
            # Build model
            # Input dim is the number of features (columns in X_train)
            input_dim = X_train.shape[1]
            classifier.build_model(input_dim)

            classifier.train(X_train, y_train, epochs=20, batch_size=32)

            classifier.save_model(model_path)

        classifier.evaluate(X_test, y_test)
    else:
        print(f"Error: Data file not found at {data_path}")


if __name__ == "__main__":
    main()
