import os
from dotenv import load_dotenv
from collections import deque
import numpy as np
import pandas as pd

load_dotenv()

class LandmarkHandler:

    """
    Handles a temporal sequence of hand landmarks frames, normalizes them,
    and prepares them for ML model input.
    """

    LANDMARK_WRIST = int(os.getenv("HAND_LANDMARK_WRIST", 0))
    LANDMARK_MIDDLE_FINGER = int(os.getenv("HAND_LANDMARK_MIDDLE_FINGER", 0))
    TOTAL_LANDMARKS = int(os.getenv("HAND_TOTAL_LANDMARKS", 0))

    def __init__(self, n_frames: int):
        """
        Creates an internal buffer to handle landmarks up to n_frames then it's ready to be
        processed by the tensor models

        For static images frames it will be expected 1, for video depends on the sequence
        
        :param n_frames: Number of frames, buffer len
        :type n_frames: int

        """
        self.buffer = deque(maxlen = n_frames)

    def add_frame(self, landmarks: np.ndarray) -> None:
        """
        Add a single frame of landmarks to the buffer.

        :param landmarks: np.ndarray of shape (21,3)

        """
        self.buffer.append(landmarks.flatten())

    def ready(self) -> bool:
        """
        Check if the buffer has reached the configured number of frames

        """
        return len(self.buffer) == self.buffer.maxlen
    
    def clear(self) ->  None:
        """
        Clear all frames from the buffer.

        """
        self.buffer.clear()
    
    def export(self) -> np.ndarray:
        """
        Export all the values raw from the buffer
        
        :return: Multidimensional array with shape (n_frames, 63)
        :rtype: ndarray[_AnyShape, dtype[Any]]

        """
        landmark_export: np.ndarray = np.array(self.buffer)
        return landmark_export
    

    # ---------------------------------------
    # (Normalization - Transformation - Preprocess) Functions
    # ---------------------------------------

    @staticmethod
    def get_landmark_cols(ndarray: np.ndarray, index: int, size: int = 3):
        # offset for (x,y,z)
        cols = index * size + np.arange(size) 

        # for all rows, only middle finger cols
        return ndarray[:, cols]
    
    @classmethod
    def tensor_landmarks(cls):
        #I don't know if I will be needing that
        pass

    @classmethod
    def _preprocess_scale(cls, landmarks_frame_data: np.ndarray) -> np.ndarray:

        landmarks_frame_data = np.array(landmarks_frame_data, copy=True)
        
        # columns for middle finger
        # get the landmark expected by index
        middle_finger = LandmarkHandler.get_landmark_cols(
            landmarks_frame_data,  
            cls.LANDMARK_MIDDLE_FINGER
        )

        # columns for wrist
        wrist_position = LandmarkHandler.get_landmark_cols(
            landmarks_frame_data,  
            cls.LANDMARK_WRIST
        )

        # euclidean distance for each row (axis = 1)
        scale = np.linalg.norm(middle_finger - wrist_position, axis=1)
        
        # in case some frame has middle = [0,0,0] divide by 0 will get an error
        # so it set a very small number to avoid an error
        scale[scale == 0] = 1e-6
        
        landmarks_frame_data = landmarks_frame_data / scale[:, np.newaxis]

        return landmarks_frame_data
    
    @classmethod
    def _preprocess_position(cls, landmarks_frame_data: np.ndarray) -> np.ndarray:
        landmarks_frame_data = np.array(landmarks_frame_data, copy=True)
        
        # columns for wrist
        wrist_positions = LandmarkHandler.get_landmark_cols(
            landmarks_frame_data,  
            cls.LANDMARK_WRIST
        )

        # subtract wrist to center hand
        # all rows
        # np.repeat is used to match both shapes
        landmarks_frame_data -= np.repeat(wrist_positions, cls.TOTAL_LANDMARKS, axis=1)        

        return landmarks_frame_data

    @classmethod
    def _preprocess_delta(cls, landmarks_frame_data: np.ndarray, cols: np.ndarray = None) -> np.ndarray:
        """
        Compute delta between frames.
        If cols is None, compute delta for all landmarks.
        If cols is provided, compute delta only for these columns.
        Concatenate delta at the end of the array.
        """
        landmarks_frame_data = np.array(landmarks_frame_data, copy=True)
        
        if cols is None:
            # all columns
            data_to_delta = landmarks_frame_data
        else:
            data_to_delta = landmarks_frame_data[:, cols]

        # compute delta
        delta = np.zeros_like(data_to_delta)
        delta[1:] = data_to_delta[1:] - data_to_delta[:-1]

        return delta
    
    @classmethod
    def _preprocess_delta_wrist(cls, landmarks_frame_data: np.ndarray) -> np.ndarray:
        """
        Compute delta only for the wrist and add at the end.
        """
        # columns for wrist
        wrist_cols = cls.LANDMARK_WRIST * 3 + np.arange(3) 
        
        # delegate to _preprocess_delta
        landmarks_frame_data = cls._preprocess_delta(landmarks_frame_data, cols=wrist_cols)
        
        return landmarks_frame_data

    @classmethod
    def preprocess_landmarks(cls, landmarks_frame_data: np.ndarray) -> np.ndarray:
        """

        Normalize and Transform values:
            - Scale (using distance from the tip middle finger and the wrist)
            - Add delta for the wirst without position normalize
            - Position (using wrist as the origin)
            - Drop wrist values
            - Add delta (change between frames) from fingers

        21:
            Depends on HAND_TOTAL_LANDMARKS

        123:
            - 20 landmarks * 3 (x,y,z) (drop wrist as always will be 0) = 60 
            - 60 * 2 (delta values subtracting wrist) = 120 
            - 120 + 3 (wrist delta)
        
        :param landmark_export: shape (n_frames, 63)
        :type landmark_export: np.ndarray[_AnyShape, dtype[Any]]
        :return: Multidimensional array with shape (n_frames, 123)
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """

        landmarks_frame_data_norm = np.array(landmarks_frame_data, copy=True)
        landmarks_frame_data_norm = cls._preprocess_scale(landmarks_frame_data_norm)
        delta_wrist = cls._preprocess_delta_wrist(landmarks_frame_data_norm)
        landmarks_frame_data_norm = cls._preprocess_position(landmarks_frame_data_norm)
        # drop wrist
        landmarks_frame_data_norm = np.delete(landmarks_frame_data_norm, [0,1,2], axis=1)
        delta_all = cls._preprocess_delta(landmarks_frame_data_norm)

        # add the delta data
        landmarks_frame_data_norm = np.hstack([landmarks_frame_data_norm, delta_all, delta_wrist])
       
        return landmarks_frame_data_norm
    

    @classmethod
    def to_dataframe(cls, landmarks_frame_data: np.ndarray) -> pd.DataFrame:
        """
        Convert a processed landmarks array into a pandas DataFrame with
        consistent column names: lmX_xyz, dX_xyz, and dw_xyz (if present).

        If the array has only 63 columns, only the lm columns are created.

        :param landmarks_frame_data: np.ndarray of shape (n_frames, n_features)
        :return: pd.DataFrame with appropriately named columns
        """
        _, n_cols = landmarks_frame_data.shape
        n_coords = 3  # x, y, z

        # If it's only the original landmarks (63 cols), create only lm columns
        if n_cols == cls.TOTAL_LANDMARKS * n_coords:
            lm_cols = [f"lm{i}_{axis}" for i in range(cls.TOTAL_LANDMARKS) for axis in ['x','y','z']]
            column_names = lm_cols
        else:
            n_landmarks = cls.TOTAL_LANDMARKS - 1  # without wrist
            # Landmark columns
            lm_cols = [f"lm{i+1}_{axis}" for i in range(n_landmarks) for axis in ['x','y','z']]
            # Delta columns
            delta_cols = [f"d{i+1}_{axis}" for i in range(n_landmarks) for axis in ['x','y','z']]
            # Wrist delta columns
            delta_wrist_cols = [f"dw_{axis}" for axis in ['x','y','z']]

            column_names = lm_cols + delta_cols + delta_wrist_cols

        if n_cols != len(column_names):
            raise ValueError(f"Shape mismatch: array has {n_cols} columns, expected {len(column_names)}")

        df_landmarks = pd.DataFrame(landmarks_frame_data, columns=column_names)
        return df_landmarks
    


if __name__ == "__main__":
    handler = LandmarkHandler(n_frames=2)

    frame_0 = np.array([
        [0.5795815, 0.73815686, -1.8038166e-07],
        [0.55215597, 0.6747151, -0.01099427],
        [0.5049775, 0.6430089, -0.02111784],
        [0.4683286, 0.6703038, -0.0297561],
        [0.45375395, 0.7237727, -0.03962269],
        [0.49235883, 0.62782544, -0.02785791],
        [0.45791447, 0.6919856, -0.04511432],
        [0.4612876, 0.7189338, -0.05663097],
        [0.4696094, 0.7235899, -0.06393994],
        [0.5109241, 0.66942954, -0.03136811],
        [0.47782707, 0.745063, -0.04599185],
        [0.48567954, 0.7604859, -0.05130501],
        [0.4981397, 0.7540408, -0.05566616],
        [0.53220457, 0.71796024, -0.03511057],
        [0.5018302, 0.78522193, -0.0497081],
        [0.50925815, 0.79420924, -0.04805692],
        [0.520422, 0.7833466, -0.04640864],
        [0.5514658, 0.7645711, -0.03927271],
        [0.5284891, 0.8183487, -0.05078094],
        [0.5353743, 0.8214972, -0.04739401],
        [0.5443849, 0.80915415, -0.0436805],
    ], dtype=np.float32)

    frame_1 = np.array([
        [0.56771237, 0.67958665, -3.7033789e-07],
        [0.54511535, 0.66276723, -0.01225006],
        [0.52102906, 0.62057233, -0.02380931],
        [0.5017718, 0.58466864, -0.03451538],
        [0.48849005, 0.56022024, -0.04429118],
        [0.56286085, 0.49751887, -0.01910602],
        [0.52913374, 0.49198934, -0.04274818],
        [0.5115025, 0.5324064, -0.06085577],
        [0.50408137, 0.5721977, -0.06990312],
        [0.56831604, 0.49835512, -0.02099565],
        [0.5321308, 0.5006625, -0.04019247],
        [0.5101404, 0.5454227, -0.04964491],
        [0.5025562, 0.5887555, -0.05415682],
        [0.56973356, 0.5147706, -0.02491779],
        [0.53236204, 0.5216817, -0.04318246],
        [0.5104225, 0.565379, -0.04597792],
        [0.5034066, 0.60528, -0.04480152],
        [0.5663274, 0.5443039, -0.02941359],
        [0.5284542, 0.5526613, -0.0433124],
        [0.51208436, 0.5831215, -0.04355562],
        [0.5073924, 0.6091327, -0.04049787],
    ], dtype=np.float32)

    handler.add_frame(frame_0)
    handler.add_frame(frame_1)

    print(f"Frame 1 added")
    print(f"Frame 2 added")

    """
    #random values
    for i in range(2):
        # shape (21, 3)
        simulated_landmarks = np.random.rand(21, 3).astype(np.float32)
        handler.add_frame(simulated_landmarks)
        print(f"Frame {i} added")
    """

    if handler.ready():

        # Export
        raw_landmarks = handler.export()
        print("Landmarks raw shape:", raw_landmarks.shape)

        processed_landmarks = handler.preprocess_landmarks(raw_landmarks)
        print("Landmarks processed shape:", processed_landmarks.shape)
 
        handler.clear()

        df_landmarks = handler.to_dataframe(processed_landmarks)

        df_landmarks.to_csv("processed_landmarks.csv", index=False)
        



