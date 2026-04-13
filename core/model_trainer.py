import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

class ModelTrainer:
    """
    Responsible for building, training and evaluating a gesture classification model
    using TensorFlow/Keras. Relies on data provided by DataLoader.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int,
        config: Dict[str, Union[float, int, str]] = None,
        model_path: str = "models/sign_lstm.h5"
    ):
        """
        Initialize the trainer.

        :param input_shape: Tuple (sequence_length, n_features).
        :param num_classes: Number of output classes.
        :param config: Dictionary containing hyperparameters (e.g., learning_rate,
                       lstm_units, dropout_rate). If None, defaults are used.
        :param model_path: Path where the trained model will be saved.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_path = model_path

        # Default configuration
        self.config = {
            'lstm_units': 128,
            'dropout_rate': 0.4,
            'learning_rate': 1e-3,
            'epochs': 50,
            'batch_size': 16,
            'patience': 10
        }
        if config is not None:
            self.config.update(config)

        self.model: Model = None
        self.history: tf.keras.callbacks.History = None

    def build_model(self) -> None:
        """
        Build the Keras Sequential model with LSTM layers.
        The architecture:
            - LSTM(128, return_sequences=True)
            - Dropout(0.4)
            - LSTM(64)
            - Dropout(0.4)
            - Dense(num_classes, activation='softmax')
        """

        """
        self.model = Sequential([
            LSTM(
                self.config['lstm_units'],
                return_sequences=True,
                input_shape=self.input_shape
            ),
            Dropout(self.config['dropout_rate']),
            LSTM(self.config['lstm_units'] // 2),  # 64 if default 128
            Dropout(self.config['dropout_rate']),
            Dense(self.num_classes, activation='softmax')
        ])
        """
        lstm_units = self.config['lstm_units']
        bidir_units = lstm_units // 2   # 64
        self.model = Sequential([
            Bidirectional(LSTM(bidir_units, return_sequences=True), input_shape=self.input_shape),
            Dropout(0.4),
            LSTM(lstm_units // 2),  # 64, ahora entrada 128
            Dropout(0.4),
            Dense(self.num_classes, activation='softmax')
        ])


        print("Model built successfully.")
        self.model.summary()

    def compile_model(
        self,
        optimizer: Optimizer = None,
        loss: Loss = None,
        metrics: List[str] = None
    ) -> None:
        """
        Compile the model with the given optimizer, loss, and metrics.
        If not provided, defaults are Adam, categorical_crossentropy, and ['accuracy'].

        :param optimizer: Keras optimizer instance. If None, uses Adam with learning_rate from config.
        :param loss: Keras loss instance. If None, uses CategoricalCrossentropy.
        :param metrics: List of metric names. If None, uses ['accuracy'].
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        if loss is None:
            loss = tf.keras.losses.CategoricalCrossentropy()
        if metrics is None:
            metrics = ['accuracy']

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print("Model compiled.")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = None,
        batch_size: int = None,
        callbacks: List[Callback] = None
    ) -> None:
        """
        Train the model.

        :param X_train: Training features.
        :param y_train: Training labels (one-hot encoded).
        :param X_val: Validation features.
        :param y_val: Validation labels (one-hot encoded).
        :param epochs: Number of epochs. If None, uses value from config.
        :param batch_size: Batch size. If None, uses value from config.
        :param callbacks: List of Keras callbacks. If None, default EarlyStopping and ModelCheckpoint are used.
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        epochs = epochs or self.config['epochs']
        batch_size = batch_size or self.config['batch_size']

        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                EarlyStopping( # avoid overfitting
                    monitor='val_loss',
                    patience=self.config['patience'],
                    restore_best_weights=True
                ),
                ModelCheckpoint( # save best model
                    self.model_path,
                    monitor='val_loss',
                    save_best_only=True
                )
            ]

        print("Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        print("Training finished.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on the test set.

        :param X_test: Test features.
        :param y_test: Test labels (one-hot encoded).
        :return: Dictionary of metric names and values (e.g., {'loss': ..., 'accuracy': ...}).
        """
        if self.model is None:
            raise RuntimeError("Model not built or trained.")

        print("Evaluating on test set...")
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {name: value for name, value in zip(self.model.metrics_names, results)}
        print(f"Test metrics: {metrics}")
        return metrics

    def save_model(self, model_path: str, encoder: LabelEncoder = None) -> None:
        """
        Save the trained model to disk. Optionally also save the label encoder classes.

        :param model_path: Path where the model will be saved (HDF5 format).
        :param encoder: Optional fitted LabelEncoder. If provided, its classes are saved alongside.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.model.save(model_path)
        print(f"Model saved to: {model_path}")

        if encoder is not None:
            encoder_path = model_path.replace('.h5', '_encoder.npy').replace('.keras', '_encoder.npy')
            np.save(encoder_path, encoder.classes_)
            print(f"Label encoder classes saved to: {encoder_path}")


if __name__ == "__main__":
    from core.data_loader import DataLoader
    from core.utils import load_env, get_project_root

    load_env()

    # Configuration
    ROOT_PATH = get_project_root() or "./"
    DATA_PATH      = os.path.join(ROOT_PATH,os.getenv("DATA_PATH", "data/processed/"))
    MODEL_OUTPUT   = os.path.join(ROOT_PATH,os.getenv("SIGN_TRANSLATE_MODEL", "models/sign_lstm.keras"))
    ENCODER_OUTPUT = os.path.join(ROOT_PATH,os.getenv("LABEL_ENCODER_FILE", "models/sign_lstm_encoder.npy"))

    # Load data
    loader = DataLoader(DATA_PATH, test_size=0.15, val_size=0.15, random_state=42)
    X, y = loader.load_data()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Classes: {loader.get_classes()}")

    # Get splits
    X_train, X_val, X_test, y_train, y_val, y_test = loader.get_splits()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 3. Build and train model
    trainer = ModelTrainer(
        input_shape=(X.shape[1], X.shape[2]),
        num_classes=loader.num_classes,
        config={'learning_rate': 1e-3, 'epochs': 50, 'batch_size': 16},
        model_path=MODEL_OUTPUT
    )
    trainer.build_model()
    trainer.compile_model()
    trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)

    # Save model and encoder
    trainer.save_model(MODEL_OUTPUT, encoder=loader.label_encoder)