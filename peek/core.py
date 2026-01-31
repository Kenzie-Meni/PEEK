import numpy as np
import cv2
from scipy.special import entr


class PEEK:
    """
    PEEK map computation using the original YOLOv5-era math.

    Given feature maps x with shape (H, W, C):

      1) Positivity shift:
           x_pos = x + abs(min(x))

      2) "Pseudo-entropy" over channels:
           peek = -sum_c entr(x_pos)_c
                = -sum_c ( -x_pos_c * log(x_pos_c) )
                =  sum_c x_pos_c * log(x_pos_c)

    Notes
    - This is intentionally not Shannon entropy: it does not normalize channels.
    - The global shift uses the global minimum over all H, W, C.
    """

    def __init__(self, eps=1e-12):
        self.eps = float(eps)

    def __call__(self, feature_maps_hwc: np.ndarray) -> np.ndarray:
        # Feature maps in HWC format
        x = feature_maps_hwc.astype(np.float32)

        # Original positivity shift (global minimum)
        x_pos = x + np.abs(np.min(x))

        # Stabilize log(0) edge cases without changing the intent
        x_pos = x_pos + self.eps

        # Original pseudo-entropy: -sum(entr(.), axis=-1)
        # entr(z) = -z * log(z)
        # so -sum(entr(z)) = sum(z * log(z))
        peek_map = -np.sum(entr(x_pos), axis=-1)

        return peek_map

    def resize(self, peek_map_hw: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        # cv2.resize expects (width, height)
        h, w = out_hw
        return cv2.resize(peek_map_hw, (h, w))
