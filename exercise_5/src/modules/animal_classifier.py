import os
import ssl
from typing import Optional, Tuple

import numpy as np
from keras import datasets, layers, models
from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau
from keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class AnimalClassifier:
    """
    A Convolutional Neural Network (CNN) classifier for the CIFAR-10 dataset.
    This dataset contains 60,000 32x32 color images in 10 classes.
    """

    def __init__(self) -> None:
        self.model: Optional[models.Sequential] = None
        # Class names for CIFAR-10
        self.class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def load_and_preprocess_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the CIFAR-10 dataset and preprocesses it.
        Preprocessing includes normalizing pixel values to be between 0 and 1.
        """
        print("Loading CIFAR-10 dataset...")

        # Workaround for SSL certificate verification failure on some systems (e.g. macOS)
        # This allows downloading the dataset without installing specific certificates.
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # The load_data() function returns tuple of numpy arrays: (x_train, y_train), (x_test, y_test)
        (train_images, train_labels), (test_images, test_labels) = (
            datasets.cifar10.load_data()
        )

        # Normalize pixel values to be between 0 and 1
        # Pixel values are integers between 0 and 255. Dividing by 255.0 converts them to floats [0, 1].
        # This helps the neural network converge faster during training.
        print("Normalizing pixel values...")
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        print(f"Training data shape: {train_images.shape}")
        print(f"Test data shape: {test_images.shape}")

        return train_images, test_images, train_labels, test_labels

    def build_model(self) -> None:
        """
        Builds a Convolutional Neural Network (CNN).
        CNNs are particularly good at finding patterns in images.
        """
        print("Building Convolutional Neural Network...")
        self.model = models.Sequential()

        # Convolutional Block 1
        # Conv2D: This layer creates a convolution kernel that is convolved with the layer input
        # to produce a tensor of outputs.
        # 32 filters of size (3, 3). relu activation introduces non-linearity.
        # input_shape=(32, 32, 3) corresponds to height, width, and color channels (RGB) of CIFAR-10 images.
        self.model.add(
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3))
        )

        # MaxPooling2D: Downsamples the input representation by taking the maximum value over the window defined by pool_size (2, 2).
        # This reduces the spatial dimensions (height, width) of the output volume.
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Convolutional Block 2
        # Increasing the number of filters to 64 to capture more complex features.
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Convolutional Block 3
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu"))

        # Flatten layer: Converts the 3D output of the last convolutional layer (height, width, filters)
        # into a 1D array to feed into the dense layers.
        self.model.add(layers.Flatten())

        # Dense Block
        # Dense layer with 64 units and ReLU activation.
        self.model.add(layers.Dense(64, activation="relu"))

        # Output Layer
        # 10 units corresponding to the 10 classes of CIFAR-10.
        # No activation function here means it outputs "logits" (raw prediction scores).
        # We could use 'softmax' to get probabilities, but using from_logits=True in loss function is often more stable.
        self.model.add(layers.Dense(10))

        # Compile the model
        # Optimizer: Adam is a standard optimizer that adjusts learning rate during training.
        # Loss: SparseCategoricalCrossentropy is used when labels are integers (0-9).
        # from_logits=True tells the loss function to apply softmax internally.
        self.model.compile(
            optimizer="adam",
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        self.model.summary()

    def train_with_augmentation(
        self, train_images: np.ndarray, train_labels: np.ndarray
    ) -> History:
        """
        Trains the model on the training data using data augmentation.
        Data augmentation generates new training samples by randomly transforming
        existing images (rotation, shift, flip, etc.). This helps reduce overfitting
        and improves the model's ability to generalize to new data.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Starting training with data augmentation...")

        # Create an ImageDataGenerator for data augmentation
        # rotation_range=15: Randomly rotate images by up to 15 degrees
        # width_shift_range=0.1: Randomly shift images horizontally by up to 10% of width
        # height_shift_range=0.1: Randomly shift images vertically by up to 10% of height
        # horizontal_flip=True: Randomly flip images horizontally
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=0.1,
        )

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            train_images, train_labels, test_size=0.1, random_state=42
        )

        # Fit the augmentation generator on the training data
        datagen.fit(X_train)

        # Stop if validation loss doesn't improve for 10 epochs
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Reduce learning rate if validation loss doesn't improve for 5 epochs
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
        )

        # Train the model using the generator
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=64),
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )
        return history

    def evaluate(
        self, test_images: np.ndarray, test_labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluates the model on unseen test data.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(test_images, test_labels, verbose=2)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def save_model(self, filepath: str) -> None:
        """
        Saves the trained model to the specified filepath.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        print(f"Saving model to {filepath}...")
        self.model.save(filepath)
        print("Model saved.")

    def load_model(self, filepath: str) -> None:
        """
        Loads a trained model from the specified filepath.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        print(f"Loading model from {filepath}...")
        self.model = models.load_model(filepath)
        print("Model loaded.")
