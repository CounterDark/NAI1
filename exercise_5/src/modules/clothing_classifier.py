import os
import ssl
from typing import Optional, Tuple

import numpy as np
from keras import datasets, layers, models
from keras.callbacks import EarlyStopping, History
from keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split


class ClothingClassifier:
    """
    A classifier for the Fashion-MNIST dataset comparing MLP and CNN architectures.
    Fashion-MNIST consists of 60,000 28x28 grayscale images of 10 fashion categories.
    """

    def __init__(self) -> None:
        self.mlp_model: Optional[models.Sequential] = None
        self.cnn_model: Optional[models.Sequential] = None
        self.class_names = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

    def load_and_preprocess_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the Fashion-MNIST dataset and preprocesses it.
        """
        print("Loading Fashion-MNIST dataset...")

        # Workaround for SSL certificate verification failure on macOS
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        (train_images, train_labels), (test_images, test_labels) = (
            datasets.fashion_mnist.load_data()
        )

        # Normalize pixel values to be between 0 and 1
        # Unlike CIFAR-10 where we used in-model Rescaling for CNN, standard MLP often expects 0-1 inputs directly.
        # Images are grayscale (28, 28), pixel values 0-255.
        train_images = train_images.astype("float32") / 255.0
        test_images = test_images.astype("float32") / 255.0

        print(f"Training data shape: {train_images.shape}")
        print(f"Test data shape: {test_images.shape}")

        return train_images, test_images, train_labels, test_labels

    def build_mlp_model(self) -> None:
        """
        Builds a simple Dense Neural Network (MLP).
        Architecture: Flatten(28x28) -> Dense(128, relu) -> Dense(10, softmax)
        """
        print("\nBuilding MLP Model...")
        self.mlp_model = models.Sequential(
            [
                layers.Flatten(
                    input_shape=(28, 28)
                ),  # Input layer: 28x28 = 784 neurons
                layers.Dense(128, activation="relu"),  # Hidden layer
                layers.Dense(10, activation="softmax"),  # Output layer: 10 classes
            ]
        )

        self.mlp_model.compile(
            optimizer="adam",
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.mlp_model.summary()

    def build_cnn_model(self) -> None:
        """
        Builds a Convolutional Neural Network (CNN).
        Architecture: Conv2D -> MaxPooling -> Flatten -> Dense -> Output
        """
        print("\nBuilding CNN Model...")
        self.cnn_model = models.Sequential(
            [
                # We need to specify input_shape as (28, 28, 1) for grayscale
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(10, activation="softmax"),
            ]
        )

        self.cnn_model.compile(
            optimizer="adam",
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.cnn_model.summary()

    def train_mlp(
        self, train_images: np.ndarray, train_labels: np.ndarray, epochs: int = 15
    ) -> History:
        """
        Trains the MLP model.
        """
        if self.mlp_model is None:
            raise ValueError("MLP Model not built. Call build_mlp_model() first.")

        print("\nStarting MLP training...")

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_images, train_labels, test_size=0.1, random_state=42
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = self.mlp_model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1,
        )
        return history

    def train_cnn(
        self, train_images: np.ndarray, train_labels: np.ndarray, epochs: int = 15
    ) -> History:
        """
        Trains the CNN model.
        Note: Expects images with channel dimension (28, 28, 1).
        """
        if self.cnn_model is None:
            raise ValueError("CNN Model not built. Call build_cnn_model() first.")

        print("\nStarting CNN training...")

        # Add channel dimension if missing
        if train_images.ndim == 3:
            train_images = np.expand_dims(train_images, axis=-1)

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_images, train_labels, test_size=0.1, random_state=42
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = self.cnn_model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1,
        )
        return history

    def evaluate_models(
        self, test_images: np.ndarray, test_labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluates both models and prints the comparison.
        Returns tuple (mlp_accuracy, cnn_accuracy).
        """
        print("\n--- Model Comparison ---")

        # Evaluate MLP
        print("Evaluating MLP...")
        if self.mlp_model is None:
            raise ValueError("MLP Model not available.")
        _, mlp_acc = self.mlp_model.evaluate(test_images, test_labels, verbose=0)
        print(f"MLP Test Accuracy: {mlp_acc:.4f}")

        # Evaluate CNN
        print("Evaluating CNN...")
        if self.cnn_model is None:
            raise ValueError("CNN Model not available.")

        # Prepare data for CNN (add channel dim)
        test_images_cnn = test_images
        if test_images.ndim == 3:
            test_images_cnn = np.expand_dims(test_images, axis=-1)

        _, cnn_acc = self.cnn_model.evaluate(test_images_cnn, test_labels, verbose=0)
        print(f"CNN Test Accuracy: {cnn_acc:.4f}")

        diff = cnn_acc - mlp_acc
        print(f"\nAccuracy Difference (CNN - MLP): {diff:.4f}")
        if diff > 0:
            print("CNN performed better.")
        elif diff < 0:
            print("MLP performed better.")
        else:
            print("Both models performed equally.")

        return mlp_acc, cnn_acc

    def save_models(self, base_path: str) -> None:
        """
        Saves both models.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        if self.mlp_model:
            mlp_path = base_path.replace(".keras", "_mlp.keras")
            print(f"Saving MLP model to {mlp_path}...")
            self.mlp_model.save(mlp_path)

        if self.cnn_model:
            cnn_path = base_path.replace(".keras", "_cnn.keras")
            print(f"Saving CNN model to {cnn_path}...")
            self.cnn_model.save(cnn_path)
