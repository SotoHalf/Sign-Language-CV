import os
import numpy as np
from typing import Optional, Tuple
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class ModelHandler:
    """
    Loads a pre-trained Keras model and its label encoder, and exposes
    prediction methods for use at inference time.
    """

    def __init__(self) -> None:
        """Initialize the handler with empty references before any model is loaded."""
        self.model: Optional[tf.keras.Model] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.input_shape: Optional[Tuple[int, int]] = None   # (sequence_length, n_features)
        self.num_classes: Optional[int] = None

    def load(self, model_path: str, encoder_path: str) -> None:
        """
        Load a trained Keras model and the associated label encoder from disk.

        The model may be in ``.h5`` or ``.keras`` format.
        The encoder must be a ``.npy`` file produced by ``np.save(path, encoder.classes_)``.

        :param model_path: Path to the saved Keras model file.
        :type model_path: str
        :param encoder_path: Path to the saved label encoder classes (``.npy``).
        :type encoder_path: str
        :raises FileNotFoundError: If either file does not exist.
        :raises ValueError: If the model output size does not match the number of encoder classes.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

        classes = np.load(encoder_path, allow_pickle=True)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = classes
        print(f"Label encoder loaded with {len(classes)} classes")

        self.input_shape = self.model.input_shape[1:]   # strip batch dimension
        self.num_classes = self.model.output_shape[-1]

        if self.num_classes != len(classes):
            raise ValueError(
                f"Model output classes ({self.num_classes}) does not match "
                f"encoder classes ({len(classes)})"
            )

    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        """
        Predict the class label for a single landmark sequence.

        :param sequence: Input array of shape ``(sequence_length, n_features)``.
        :type sequence: np.ndarray
        :return: Tuple of ``(predicted_label, confidence)`` where confidence is
            the softmax probability of the predicted class.
        :rtype: tuple[str, float]
        :raises RuntimeError: If the model has not been loaded yet.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        # Add batch dimension → (1, seq_len, n_features)
        probs = self.model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        idx: int = int(np.argmax(probs))
        predicted_label: str = self.label_encoder.inverse_transform([idx])[0]

        return predicted_label, float(probs[idx])

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """
        Return the full softmax probability vector for a single sequence.

        :param sequence: Input array of shape ``(sequence_length, n_features)``.
        :type sequence: np.ndarray
        :return: Probability array of shape ``(num_classes,)``.
        :rtype: np.ndarray
        :raises RuntimeError: If the model has not been loaded yet.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

    def is_loaded(self) -> bool:
        """
        Check whether both the model and label encoder have been loaded.

        :return: ``True`` if the handler is ready for inference.
        :rtype: bool
        """
        return self.model is not None and self.label_encoder is not None
