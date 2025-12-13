import os
import ssl
from typing import Optional, Tuple

import numpy as np
from keras import datasets, layers, models
from keras.callbacks import EarlyStopping, History
from keras.losses import BinaryCrossentropy
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class IMDBReviewSentimentClassifier:
    """
    A classifier for the IMDB Movie Review dataset using an RNN (LSTM).
    Classifies reviews as positive (1) or negative (0).
    """

    def __init__(self, vocab_size: int = 10000, max_length: int = 200) -> None:
        self.model: Optional[models.Sequential] = None
        self.vocab_size = vocab_size
        self.max_length = max_length

    def load_and_preprocess_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the IMDB dataset and preprocesses it.
        Reviews are sequences of integers (word indices).
        """
        print("Loading IMDB Movie Review dataset...")

        # Workaround for SSL certificate verification failure on macOS
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Only keep the top 'vocab_size' most frequent words
        (train_data, train_labels), (test_data, test_labels) = datasets.imdb.load_data(
            num_words=self.vocab_size
        )

        print(f"Training data size: {len(train_data)}")
        print(f"Test data size: {len(test_data)}")

        # Pad sequences to ensure uniform length
        print(f"Padding sequences to max length {self.max_length}...")
        train_data = pad_sequences(train_data, maxlen=self.max_length)
        test_data = pad_sequences(test_data, maxlen=self.max_length)

        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")

        return train_data, test_data, train_labels, test_labels

    def build_model(self) -> None:
        """
        Builds a Recurrent Neural Network (RNN) with Embedding and LSTM layers.
        Architecture: Embedding -> LSTM -> Dense(ReLU) -> Dense(Sigmoid)
        """
        print("Building RNN (LSTM) Model...")
        self.model = models.Sequential()

        # Embedding Layer: Converts integer word indices into dense vectors of fixed size.
        # input_dim: Size of the vocabulary (vocab_size)
        # output_dim: Dimension of the dense embedding
        # input_length: Length of input sequences
        self.model.add(
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=128,  # , input_length=self.max_length
            )
        )

        # LSTM Layer: Long Short-Term Memory layer to process sequences.
        # 64 units
        # dropout: Fraction of units to drop for the linear transformation of the inputs
        # recurrent_dropout: Fraction of units to drop for the linear transformation of the recurrent state
        self.model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))

        # Dense Layer: Fully connected layer
        self.model.add(layers.Dense(64, activation="relu"))

        # Dropout to prevent overfitting
        self.model.add(layers.Dropout(0.5))

        # Output Layer: Binary classification (Positive/Negative)
        self.model.add(layers.Dense(1, activation="sigmoid"))

        self.model.compile(
            optimizer="adam",
            loss=BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        self.model.summary()

    def train(
        self, train_data: np.ndarray, train_labels: np.ndarray, epochs: int = 10
    ) -> History:
        """
        Trains the RNN model.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Starting training...")

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_labels, test_size=0.2, random_state=42
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=64,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1,
        )
        return history

    def evaluate(
        self, test_data: np.ndarray, test_labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluates the model on test data.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(test_data, test_labels, verbose=2)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def save_model(self, filepath: str) -> None:
        """
        Saves the trained model to the specified filepath.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

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
