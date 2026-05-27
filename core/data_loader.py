import os
import numpy as np
import pandas as pd
from glob import glob
from typing import List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


class DataLoader:
    """
    Loads and preprocesses the landmark dataset from a folder structure.

    Each subfolder represents a class label and contains ``.csv`` files,
    where each file is one recorded gesture sequence.

    Expected layout::

        dataset_path/
            a/
                <uid>.csv
                <uid>.csv
            b/
                <uid>.csv
    """

    def __init__(
        self,
        dataset_path: str,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> None:
        """
        Configure the loader with split ratios.

        :param dataset_path: Root folder containing one subfolder per class label.
        :type dataset_path: str
        :param test_size: Fraction of samples to reserve for the test set.
        :type test_size: float
        :param val_size: Fraction of samples to reserve for the validation set.
        :type val_size: float
        :param random_state: Random seed for reproducible splits.
        :type random_state: int
        """
        self.dataset_path: str = dataset_path
        self.test_size: float = test_size
        self.val_size: float = val_size
        self.random_state: int = random_state

        self.label_encoder: LabelEncoder = LabelEncoder()
        self.num_classes: Optional[int] = None
        self.sequence_length: Optional[int] = None
        self.n_features: Optional[int] = None

        self.X: Optional[np.ndarray] = None   # (n_samples, seq_len, n_features)
        self.y: Optional[np.ndarray] = None   # (n_samples,) string labels

        self._y_encoded: Optional[np.ndarray] = None   # integer class indices
        self._y_onehot: Optional[np.ndarray] = None    # one-hot encoded labels

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan the dataset folder and load all CSV sequences into memory.

        Each CSV must contain landmark feature columns (and optionally a ``Frame``
        column which is dropped). All CSVs in the same dataset must have the same
        number of rows (frames) and columns (features).

        :return: Tuple ``(X, y)`` where ``X`` has shape ``(n_samples, seq_len, n_features)``
            and ``y`` is an array of string class labels with shape ``(n_samples,)``.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        X_list: List[np.ndarray] = []
        y_list: List[str] = []

        for label in sorted(os.listdir(self.dataset_path)):
            label_path = os.path.join(self.dataset_path, label)
            if not os.path.isdir(label_path):
                continue

            for csv_path in glob(os.path.join(label_path, "*.csv")):
                df = pd.read_csv(csv_path)
                if 'Frame' in df.columns:
                    df = df.drop(columns=['Frame'])

                X_list.append(df.values.astype(np.float32))
                y_list.append(label)

        self.X = np.array(X_list)
        self.y = np.array(y_list)

        self.sequence_length = self.X.shape[1]
        self.n_features = self.X.shape[2]

        self._y_encoded = self.label_encoder.fit_transform(self.y)
        self._y_onehot = to_categorical(self._y_encoded)
        self.num_classes = self._y_onehot.shape[1]

        return self.X, self.y

    def get_splits(self) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Split the loaded data into training, validation and test sets.

        The first split is stratified to preserve class distribution.
        Labels returned are one-hot encoded.

        :return: ``(X_train, X_val, X_test, y_train, y_val, y_test)``
        :rtype: tuple of np.ndarray
        :raises RuntimeError: If :meth:`load_data` has not been called yet.
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # First split: isolate training set, keep (val + test) together
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X,
            self._y_onehot,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
            stratify=self._y_encoded
        )

        # Second split: divide the remainder into val and test
        val_ratio: float = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - val_ratio,
            random_state=self.random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_classes(self) -> List[str]:
        """
        Return the list of class names as inferred by the label encoder.

        :return: Alphabetically sorted list of class label strings.
        :rtype: list[str]
        :raises RuntimeError: If :meth:`load_data` has not been called yet.
        """
        if self.label_encoder.classes_.size == 0:
            raise RuntimeError("Encoder not fitted. Call load_data() first.")
        return self.label_encoder.classes_.tolist()

    def save_encoder(self, encoder_path: str) -> None:
        """
        Persist the fitted label encoder classes to a ``.npy`` file.

        :param encoder_path: Destination file path.
        :type encoder_path: str
        :raises RuntimeError: If :meth:`load_data` has not been called yet.
        """
        if self.label_encoder.classes_.size == 0:
            raise RuntimeError("Encoder not fitted. Call load_data() first.")
        np.save(encoder_path, self.label_encoder.classes_)
        print(f"[INFO] Label encoder classes saved to: {encoder_path}")
