import numpy as np
from ..base import BaseFilter

"""
ChatGPT Usage:
The [0.2989, 0.5870, 0.1140] values that are used instead of [1/3, 1/3, 1/3]
"""

class Saturation(BaseFilter):
    def __init__(self, alpha: float = 1.0):
        """
        Saturation Filter.
        Blends the image with a grayscale version to reduce or restore color intensity.

        Parameters:
            alpha (float): value between [0, 30], 0 = full grayscale, 1 = original color.
        """
        if not (0 <= alpha <= 30):
            raise ValueError(f"Alpha must be in [0, 30], got {alpha}")

        self.alpha = alpha

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.shape[2] != 3:
            raise ValueError("Saturation adjustment requires an RGB image.")

        # Computing grayscale using weighted sum (according to our light sensitivity)
        gray = np.dot(image, [0.2989, 0.5870, 0.1140])[:, :, np.newaxis]
        return gray * (1 - self.alpha) + image * self.alpha

