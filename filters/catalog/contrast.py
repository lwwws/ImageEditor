import numpy as np
from ..base import BaseFilter

class Contrast(BaseFilter):
    def __init__(self, alpha: float = 1.0):
        """
        Contrast Filter.
        Adjusts contrast by stretching or compressing pixel values around the mean.

        Parameters:
            factor (float): >1 increases contrast, <1 decreases. 1 = no change.
        """
        if not (-5 <= alpha <= 5):
            raise ValueError(f"Alpha must be in [-5, 5], got {alpha}")

        self.alpha = alpha

    def apply(self, image: np.ndarray) -> np.ndarray:
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return (image - mean) * self.alpha + mean
