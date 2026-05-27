import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Bidirectional,
    LSTM, LayerNormalization, Dropout, GlobalAveragePooling1D, Dense
)
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


class ModelTrainer:
    """
    Builds, compiles, trains and evaluates the gesture classification model.

    Architecture: Conv1D → BiLSTM × 2 → GlobalAvgPool → Dense.
    Relies on data splits produced by :class:`~core.data_loader.DataLoader`.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int,
        config: Optional[Dict[str, Union[float, int, str]]] = None,
        model_path: str = "models/sign_lstm.h5"
    ) -> None:
        """
        Initialize the trainer with architecture and training hyperparameters.

        :param input_shape: ``(sequence_length, n_features)`` as expected by the model input.
        :type input_shape: tuple[int, int]
        :param num_classes: Number of gesture classes for the output softmax layer.
        :type num_classes: int
        :param config: Dictionary overriding any default hyperparameters.
            Valid keys: ``lstm_units``, ``dropout_rate``, ``learning_rate``,
            ``epochs``, ``batch_size``, ``patience``.
        :type config: dict, optional
        :param model_path: Destination path for saving the best checkpoint during training.
        :type model_path: str
        """
        self.input_shape: Tuple[int, int] = input_shape
        self.num_classes: int = num_classes
        self.model_path: str = model_path

        self.config: Dict[str, Union[float, int]] = {
            'lstm_units': 128,
            'dropout_rate': 0.4,
            'learning_rate': 1e-3,
            'epochs': 50,
            'batch_size': 16,
            'patience': 10,
        }
        if config is not None:
            self.config.update(config)

        self.model: Optional[Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None

    def build_model(self) -> None:
        """
        Construct the Keras model graph.

        Architecture:
            Conv1D(64) → BatchNorm → Dropout(0.2) →
            BiLSTM(lstm_units) → LayerNorm → Dropout →
            BiLSTM(lstm_units//2) → LayerNorm →
            GlobalAveragePooling1D →
            Dense(128, relu) → Dropout →
            Dense(num_classes, softmax)
        """
        lstm_units: int = self.config['lstm_units']
        dropout_rate: float = self.config['dropout_rate']

        inputs = Input(shape=self.input_shape)

        x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)

        x = Bidirectional(LSTM(lstm_units // 2, return_sequences=True))(x)
        x = LayerNormalization()(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(dropout_rate)(x)

        outputs = Dense(self.num_classes, activation="softmax")(x)

        self.model = Model(inputs, outputs)

        print("Model built successfully.")
        self.model.summary()

    def compile_model(
        self,
        optimizer: Optional[Optimizer] = None,
        loss: Optional[Loss] = None,
        metrics: Optional[List[str]] = None
    ) -> None:
        """
        Compile the model with optimizer, loss and evaluation metrics.

        Defaults: AdamW (weight_decay=1e-4) + CategoricalCrossentropy
        (label_smoothing=0.05) + accuracy.

        :param optimizer: Keras optimizer instance. Defaults to AdamW with
            the ``learning_rate`` from the config.
        :type optimizer: Optimizer, optional
        :param loss: Keras loss instance. Defaults to CategoricalCrossentropy
            with label smoothing.
        :type loss: Loss, optional
        :param metrics: List of metric names. Defaults to ``['accuracy']``.
        :type metrics: list[str], optional
        :raises RuntimeError: If :meth:`build_model` has not been called yet.
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        if optimizer is None:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.config['learning_rate'],
                weight_decay=1e-4
            )
        if loss is None:
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
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
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        callbacks: Optional[List[Callback]] = None
    ) -> None:
        """
        Train the model on the provided data splits.

        :param X_train: Training features of shape ``(n_samples, seq_len, n_features)``.
        :type X_train: np.ndarray
        :param y_train: One-hot encoded training labels of shape ``(n_samples, n_classes)``.
        :type y_train: np.ndarray
        :param X_val: Validation features, same shape convention as ``X_train``.
        :type X_val: np.ndarray
        :param y_val: One-hot encoded validation labels.
        :type y_val: np.ndarray
        :param epochs: Number of training epochs. Overrides config if provided.
        :type epochs: int, optional
        :param batch_size: Mini-batch size. Overrides config if provided.
        :type batch_size: int, optional
        :param callbacks: Keras callbacks list. Defaults to EarlyStopping +
            ModelCheckpoint monitoring ``val_loss``.
        :type callbacks: list[Callback], optional
        :raises RuntimeError: If :meth:`build_model` has not been called yet.
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        epochs = epochs or self.config['epochs']
        batch_size = batch_size or self.config['batch_size']

        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['patience'],
                    restore_best_weights=True
                ),
                ModelCheckpoint(
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
        Evaluate the trained model on the test set.

        :param X_test: Test features of shape ``(n_samples, seq_len, n_features)``.
        :type X_test: np.ndarray
        :param y_test: One-hot encoded test labels.
        :type y_test: np.ndarray
        :return: Dictionary mapping metric names to their values (e.g. ``{'loss': 0.1, 'accuracy': 0.95}``).
        :rtype: dict[str, float]
        :raises RuntimeError: If no model has been built or trained.
        """
        if self.model is None:
            raise RuntimeError("Model not built or trained.")

        print("Evaluating on test set...")
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics: Dict[str, float] = dict(zip(self.model.metrics_names, results))
        print(f"Test metrics: {metrics}")
        return metrics

    def save_model(self, model_path: str, encoder: Optional[LabelEncoder] = None) -> None:
        """
        Save the trained model to disk, and optionally the label encoder classes.

        The encoder classes are saved as ``<model_stem>_encoder.npy`` alongside the model.

        :param model_path: Destination path (``.keras`` or ``.h5``).
        :type model_path: str
        :param encoder: Fitted ``LabelEncoder`` whose classes will be persisted.
        :type encoder: LabelEncoder, optional
        :raises RuntimeError: If no model has been trained yet.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")

        if encoder is not None:
            encoder_path = model_path.replace('.h5', '_encoder.npy').replace('.keras', '_encoder.npy')
            np.save(encoder_path, encoder.classes_)
            print(f"Label encoder classes saved to: {encoder_path}")


if __name__ == "__main__":
    from core.data_loader import DataLoader
    from core.utils import AppPaths
    import os

    AppPaths.load_env()

    DATA_PATH     = AppPaths.path(os.getenv("DATA_PATH", "data/processed"))
    MODEL_OUTPUT  = AppPaths.path(os.getenv("SIGN_TRANSLATE_MODEL", "models/sign_lstm.keras"))
    ENCODER_OUTPUT = AppPaths.path(os.getenv("LABEL_ENCODER_FILE", "models/sign_lstm_encoder.npy"))

    loader = DataLoader(DATA_PATH, test_size=0.15, val_size=0.15, random_state=42)
    X, y = loader.load_data()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Classes: {loader.get_classes()}")

    X_train, X_val, X_test, y_train, y_val, y_test = loader.get_splits()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    trainer = ModelTrainer(
        input_shape=(X.shape[1], X.shape[2]),
        num_classes=loader.num_classes,
        config={'learning_rate': 1e-3, 'epochs': 50, 'batch_size': 16},
        model_path=MODEL_OUTPUT
    )
    trainer.build_model()
    trainer.compile_model()
    trainer.train(X_train, y_train, X_val, y_val)

    metrics = trainer.evaluate(X_test, y_test)
    trainer.save_model(MODEL_OUTPUT, encoder=loader.label_encoder)
