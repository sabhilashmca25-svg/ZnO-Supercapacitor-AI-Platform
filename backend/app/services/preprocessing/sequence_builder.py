"""
sequence_builder.py — Converts a flat feature DataFrame into a
padded sequence tensor suitable for LSTM/GRU inference.
Mirrors the exact logic used in Notebooks 05 and 06.
"""
import numpy as np
import pandas as pd
from typing import Tuple

MAX_SEQ_LEN = 651   # matches training max_seq_len


def build_padded_sequence(features: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """
    Returns:
        seq:        np.ndarray of shape (1, MAX_SEQ_LEN, n_features), dtype float32
        actual_len: int — number of real (non-padded) timesteps
    """
    arr = features.values.astype(np.float32)
    L   = len(arr)
    seq = np.zeros((1, MAX_SEQ_LEN, arr.shape[1]), dtype=np.float32)
    seq[0, :L, :] = arr
    return seq, L
