import os
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import History  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.models import Sequential, load_model  # type: ignore


class MushroomClassifier:
    def __init__(self) -> None:
        self.model: Optional[Sequential] = None
        self.columns = [
            "poisonous",
            "cap-shape",
            "cap-surface",
            "cap-color",
            "bruises",
            "odor",
            "gill-attachment",
            "gill-spacing",
            "gill-size",
            "gill-color",
            "stalk-shape",
            "stalk-root",
            "stalk-surface-above-ring",
            "stalk-surface-below-ring",
            "stalk-color-above-ring",
            "stalk-color-below-ring",
            "veil-type",
            "veil-color",
            "ring-number",
            "ring-type",
            "spore-print-color",
            "population",
            "habitat",
        ]

    def load_and_preprocess_data(
        self, filepath: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Loads data from CSV, performs One-Hot Encoding and Label Encoding,
        and splits into training and testing sets.
        """
        print(f"Loading data from {filepath}...")
        # Load dataset
        df = pd.read_csv(filepath, header=None, names=self.columns)

        # Separate target and features
        X = df.drop("poisonous", axis=1)
        y = df["poisonous"]

        # Encode target labels (p -> 1, e -> 0)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        # Check mapping
        mapping = dict(
            zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
        )
        print(f"Target mapping: {mapping}")

        # One-Hot Encoding for categorical features
        print("Preprocessing features with One-Hot Encoding...")
        X_encoded = pd.get_dummies(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42
        )

        # Convert to float32 for TensorFlow
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")
        y_train = y_train.astype("float32")
        y_test = y_test.astype("float32")

        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def build_model(self, input_dim: int) -> None:
        """
        Builds a Sequential MLP model.
        """
        print("Building Neural Network...")
        self.model = Sequential(
            [
                # Input layer is implicit in the first Dense layer via input_shape
                # Hidden Layer 1: Dense layer with ReLU activation
                Dense(32, activation="relu", input_shape=(input_dim,)),
                # Hidden Layer 2: Another Dense layer with ReLU activation
                Dense(16, activation="relu"),
                # Output Layer: 1 neuron with sigmoid activation is standard for binary classification
                # (produces a probability between 0 and 1)
                Dense(1, activation="sigmoid"),
            ]
        )

        # Compile model
        # Loss: binary_crossentropy
        # Optimizer: adam
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        self.model.summary()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        epochs: int = 20,
        batch_size: int = 32,
    ) -> History:
        """
        Trains the model.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Starting training...")
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
        )
        return history

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
        """
        Evaluates the model on test data.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
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
        self.model = load_model(filepath)
        print("Model loaded.")
