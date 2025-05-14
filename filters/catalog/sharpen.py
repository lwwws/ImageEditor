import numpy as np
from ..base import BaseFilter
from .box import Box

class Sharpen(BaseFilter):
    def __init__(self, alpha: float = 1.0, keep_dims=True, **kwargs):
        """
        Sharpen Filter.
        Enhances edges by subtracting a blurred version of the image (unsharp masking).

        Parameters:
            alpha (float): Sharpening strength, between [0, 5].
        """

        if not (0 <= alpha <= 5):
            raise ValueError(f"Alpha must be in [0, 5], got {alpha}")

        self.alpha = alpha
        self.blur = Box(height=5, width=5, keep_dims=keep_dims, **kwargs)

    def apply(self, image: np.ndarray) -> np.ndarray:
        blurred = self.blur.apply(image)
        mask = image - blurred
        sharpened = image + self.alpha * mask

        return sharpened