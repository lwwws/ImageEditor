"""
StaticFilter module
--------------------
Implements a convolutional filter using fixed kernels.

Supports applying one or more user-defined static kernels.
Inherits padding, stride, and clipping logic from ConvFilter.

Author: lwwws

ChatGPT Usage:
Used it for kernel validation, and for convolving optimizations (using sliding window view, einsum)
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Union

from .conv_filter import ConvFilter
class StaticFilter(ConvFilter):
    """
    Convolutional filter that uses fixed user-defined kernels.

    Applies one or more kernels to an image. Kernels must have shape (H, W, C),
    where C is either 1 (applied to all channels) or 3 (per-channel).
    """

    def __init__(self, kernels: Union[np.ndarray, list[np.ndarray]], **kwargs):
        """
        Initializes the static filter with one or more kernels.

        Parameters:
            kernels (np.ndarray or list[np.ndarray]): One or more 3D kernels of shape (H, W, C).
            kwargs: Passed to ConvFilter (e.g. stride, pad_val, bias, ...).
        """
        if isinstance(kernels, np.ndarray):
            kernels = [kernels]

        self._validate_kernels(kernels)
        self.kernels = kernels

        kh, kw = self.kernels[0].shape[:2]
        expected_radius = (kh // 2, kw // 2)

        if 'kernel_radius' not in kwargs:
            kwargs['kernel_radius'] = expected_radius

        if kwargs['kernel_radius'] != expected_radius:
            raise ValueError(f"Provided kernel_radius {kwargs['kernel_radius']} does not match expected radius (inferred from kernel shape): {expected_radius}")

        super().__init__(**kwargs)

    def _apply_single_filter(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Applies a single kernel to the image using sliding windows and einsum.

        Parameters:
            image (np.ndarray): Input image of shape (H, W, C)
            kernel (np.ndarray): Kernel of shape (kh, kw, 1) or (kh, kw, 3)

        Returns:
            np.ndarray: Filtered image
        """

        windows = sliding_window_view(
            image,
            window_shape=(self.kernel_height, self.kernel_width),
            axis=(0, 1)
        )

        windows = windows[::self.stride_y, ::self.stride_x]

        if kernel.shape[-1] == 1:
            kernel = np.broadcast_to(kernel, (self.kernel_height, self.kernel_width, image.shape[-1]))

        out = np.einsum('hwckl,klc->hwc', windows, kernel) + self.bias
        return np.clip(out, 0, 255)

    def convolve(self, image: np.ndarray) -> np.ndarray:
        """
        Applies all kernels sequentially to the image.

        Parameters:
            image (np.ndarray): Padded input image

        Returns:
            np.ndarray: Final output after applying all kernels
        """

        for kernel in self.kernels:
            image = self._apply_single_filter(image, kernel)

        return image

    @staticmethod
    def _validate_kernels(kernels):
        """
        Validates that all kernels have consistent shape and valid channel counts.

        Parameters:
            kernels (list[np.ndarray]): Kernels to validate

        Raises:
            ValueError if kernel shapes mismatch, are not 3D, or have invalid channels
        """

        # First: ensure self.kernels is a list of arrays
        if not (isinstance(kernels, list) and all(isinstance(k, np.ndarray) for k in kernels)):
            raise TypeError("kernels must be a numpy array or list of numpy arrays")

        # Try to extract shape from the first kernel
        try:
            first_kernel = kernels[0]
            if first_kernel.ndim != 3:
                raise ValueError(f"Unsupported kernel shape: {first_kernel.shape}, expected (H, W, C)")
            expected_shape = first_kernel.shape[:2]
        except (IndexError, AttributeError):
            raise ValueError("At least one valid 3D kernel is required")

        kh, kw = expected_shape
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError(f"Kernel shape {kh}x{kw} must have odd height and width")

        # Now validate all kernels
        for kernel in kernels:
            if kernel.ndim != 3:
                raise ValueError(f"Unsupported kernel shape: {kernel.shape}, expected (H, W, C)")
            if kernel.shape[:2] != expected_shape:
                raise ValueError(f"Kernel shape {kernel.shape[:2]} does not match expected {expected_shape}")
            if kernel.shape[2] not in (1, 3):
                raise ValueError(f"Unsupported kernel channel count: {kernel.shape[2]}")

