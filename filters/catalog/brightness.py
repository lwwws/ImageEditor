import numpy as np
from ..base import BaseFilter

class Brightness(BaseFilter):
    def __init__(self, alpha: float = 1.0):
        """
        Brightness filter.
        Adds or subtracts a constant value from all pixels.

        Parameters:
            alpha (float): Amount to brighten (positive) or darken (negative), typically ∈ [-100, 100].
        """

        if not (-100 <= alpha <= 100):
            raise ValueError(f"Alpha must be in [-100, 100], got {alpha}")

        self.alpha = alpha

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image + self.alpha
