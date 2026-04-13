import os
import numpy as np
import pandas as pd
from glob import glob
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class DataLoader:
    """
    Handles loading and preprocessing of the landmark dataset from a folder structure.
    Each subfolder represents a class label and contains CSV files of sequences.
    Prepares the data for training (train/val/test split, label encoding).
    """

    def __init__(
        self,
        dataset_path: str,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize the DataLoader.

        :param dataset_path: Path to the root folder containing class subfolders.
        :param test_size: Proportion of data to use for testing.
        :param val_size: Proportion of data to use for validation.
        :param random_state: Seed for reproducible splits.
        """
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # Internal attributes
        self.label_encoder = LabelEncoder()
        self.num_classes: int = None
        self.sequence_length: int = None
        self.n_features: int = None
        self.X: np.ndarray = None          # Raw sequences (n_samples, T, n_features)
        self.y: np.ndarray = None          # Original string labels (n_samples,)

        self._y_encoded: np.ndarray = None # Encoded integer labels
        self._y_onehot: np.ndarray = None  # One-hot encoded labels

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all sequences from the dataset folder.

        Expects a directory structure:
            dataset_path/
                class_1/
                    seq1.csv
                    seq2.csv
                class_2/
                    ...

        Each CSV file should have a 'Frame' column (which is dropped) and
        landmark columns (123 features). All sequences must have the same
        number of time steps (rows).

        :return: Tuple (X, y) where X is a numpy array of shape
                 (n_samples, sequence_length, n_features) and y is a numpy array
                 of string labels (n_samples,).
        """
        X_list = []
        y_list = []

        for label in sorted(os.listdir(self.dataset_path)):
            label_path = os.path.join(self.dataset_path, label)
            if not os.path.isdir(label_path):
                continue

            for csv_path in glob(os.path.join(label_path, "*.csv")):
                df = pd.read_csv(csv_path)

                # drop the Frame column if present
                if 'Frame' in df.columns:
                    df = df.drop(columns=['Frame'])

                # Convert to float32 numpy array (T, n_features)
                sequence = df.values.astype(np.float32)

                X_list.append(sequence)
                y_list.append(label)

        # Stack all sequences
        self.X = np.array(X_list)
        self.y = np.array(y_list)

        # Get infer dimensions
        self.sequence_length = self.X.shape[1]
        self.n_features = self.X.shape[2]

        # Encode labels
        self._y_encoded = self.label_encoder.fit_transform(self.y)
        self._y_onehot = to_categorical(self._y_encoded)
        self.num_classes = self._y_onehot.shape[1]

        return self.X, self.y

    def get_splits(self) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Split the data into training, validation and test sets.
        Uses stratified split based on the encoded labels.
        The returned y sets are one-hot encoded.

        :return: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # First split: training vs (validation + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X,
            self._y_onehot,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
            stratify=self._y_encoded
        )

        # Second split: validation vs test
        val_ratio = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - val_ratio,
            random_state=self.random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_classes(self) -> List[str]:
        """
        Get the list of class names as inferred by the LabelEncoder.

        :return: List of class names (strings).
        """
        if self.label_encoder.classes_.size == 0:
            raise RuntimeError("Encoder not fitted. Call load_data() first.")
        return self.label_encoder.classes_.tolist()

    def save_encoder(self, encoder_path: str) -> None:
        """
        Save the fitted label encoder classes to a .npy file.

        :param encoder_path: Path where the encoder classes will be saved.
        """
        if self.label_encoder.classes_.size == 0:
            raise RuntimeError("Encoder not fitted. Call load_data() first.")
        np.save(encoder_path, self.label_encoder.classes_)
        print(f"[INFO] Label encoder classes saved to: {encoder_path}")
