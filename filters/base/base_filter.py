"""
BaseFilter Module
-----------------
Defines the abstract base class for all filters.
All filters should inherit from this and implement the 'apply()' method.

Author: lwwws
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseFilter(ABC):
    """
    Abstract base class for all image filters.

    Subclasses must implement the 'apply()' method.
    This class handles general image shape handling and type restoration.
    """
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the filter to an image, handling shape and clipping.

        Parameters:
            image (np.ndarray): Input image (H, W) or (H, W, C).

        Returns:
            np.ndarray: Filtered image with same dtype and shape as input.
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        elif image.ndim != 3:
            raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")

        result = self.apply(image.astype(np.int32))
        result = np.clip(result, 0, 255)

        return result.squeeze(-1) if result.shape[-1] == 1 else result.astype(image.dtype)

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Subclasses must implement this method.

        Parameters:
            image (np.ndarray): Input image with shape (H, W, C)

        Returns:
            np.ndarray: Transformed image (H, W, C)
        """
        pass


