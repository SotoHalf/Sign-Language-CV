import os
import numpy as np
from typing import Tuple
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class ModelHandler:
    """
    Handles loading a pre-trained model and its label encoder, and provides
    prediction capabilities on new sequences.
    """

    def __init__(self):
        """Initialize the handler with empty references."""
        self.model: tf.keras.Model = None
        self.label_encoder: LabelEncoder = None
        self.input_shape: Tuple[int, int] = None
        self.num_classes: int = None

    def load(self, model_path: str, encoder_path: str) -> None:
        """
        Load a trained Keras model and the associated label encoder.

        The model can be in .h5 or .keras format
        The encoder should be a .npy file containing the classes array.

        :param model_path: Path to the saved model file.
        :param encoder_path: Path to the saved label encoder classes (.npy).
        """
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

        # Load model (supports both .h5 and .keras)
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

        # Load encoder classes
        classes = np.load(encoder_path, allow_pickle=True)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = classes
        print(f"Label encoder loaded with {len(classes)} classes")

        # Infer input shape and number of classes
        self.input_shape = self.model.input_shape[1:]  # (sequence_length, n_features)
        self.num_classes = self.model.output_shape[-1]

        # Validate consistency
        if self.num_classes != len(classes):
            raise ValueError(
                f"Model output classes ({self.num_classes}) does not match encoder classes ({len(classes)})"
            )

    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        """
        Predict the class label for a given sequence.

        :param sequence: Input sequence of shape (sequence_length, n_features).
        :return: Tuple (predicted_label, confidence).
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        # Add batch dimension (1, seq_len, n_features)
        input_batch = np.expand_dims(sequence, axis=0)

        # Get probabilities
        probs = self.model.predict(input_batch, verbose=0)[0]

        # Get class index with highest probability
        idx = np.argmax(probs)
        confidence = probs[idx]
        predicted_label = self.label_encoder.inverse_transform([idx])[0]

        return predicted_label, float(confidence)

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """
        Get class probabilities for the given sequence.

        :param sequence: Input sequence of shape (sequence_length, n_features).
        :return: Probability array of shape (num_classes,).
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        input_batch = np.expand_dims(sequence, axis=0)
        return self.model.predict(input_batch, verbose=0)[0]

    def is_loaded(self) -> bool:
        """Check if a model has been loaded successfully."""
        return self.model is not None and self.label_encoder is not None