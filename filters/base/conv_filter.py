"""
ConvFilter Module
-----------------
Provides an abstract base for convolution-based filters.

Handles kernel radius, stride, padding, and common pre/post-processing
such as padding and clipping. Subclasses must implement 'convolve()'.

Author: lwwws

ChatGPT Usage:
I used it to validate params, pad image / formula keep original dims
"""

from abc import abstractmethod
import numpy as np
import warnings
from .base_filter import BaseFilter

class ConvFilter(BaseFilter):
    """
    Abstract base class for convolution kernels.

    Subclasses must implement 'convolve()' method.
    """
    def __init__(self,
                 kernel_radius=(1, 1),
                 stride=(1, 1),
                 pad=None,
                 pad_val=0.0,
                 bias=0.0,
                 keep_dims=False):
        self._validate_params(kernel_radius,
                              stride,
                              pad,
                              pad_val,
                              bias
                              )
        """
        Initializes convolution parameters.

        Parameters:
            kernel_radius (tuple): (radius_y, radius_x), half the kernel size.
            stride (tuple): Step size in (y, x) direction.
            pad (tuple or None): Optional explicit padding. If None, computed automatically.
            pad_val (float): Value to pad with (default 0.0).
            bias (float): Value to add after convolution.
            keep_dims (bool): If True, pads image to preserve original size.
        """

        self.radius_y, self.radius_x = kernel_radius
        self.kernel_height = 2 * self.radius_y + 1
        self.kernel_width = 2 * self.radius_x + 1

        self.stride_y, self.stride_x = stride
        self.pad_val = pad_val
        self.bias = bias
        self.keep_dims_with_pad = keep_dims

        if pad is not None:
            if keep_dims:
                warnings.warn("'pad' was provided explicitly; 'keep_dims_with_pad' will be ignored.")
            self.pad_y, self.pad_x = pad
        else:
            self.pad_y = self.pad_x = None  # to be computed later

    @abstractmethod
    def convolve(self, padded_image: np.ndarray) -> np.ndarray:
        """
        Apply convolution logic to the padded image.

        Must be implemented by subclasses.

        Parameters:
            padded_image (np.ndarray): Already-padded input image of shape (H', W', C)

        Returns:
            np.ndarray: Convolved output image
        """
        pass

    def pad_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pads the input image using constant padding.

        Parameters:
            image (np.ndarray): Input image of shape (H, W, C)

        Returns:
            np.ndarray: Padded image
        """
        return np.pad(
            image,
            ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x), (0, 0)),
            mode='constant',
            constant_values=self.pad_val
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applies padding and invokes the convolution logic.

        Parameters:
            image (np.ndarray): Input image of shape (H, W, C)

        Returns:
            np.ndarray: Convolved and clipped output image
        """

        if self.pad_y is None and self.pad_x is None:
            if self.keep_dims_with_pad:
                H, W = image.shape[:2]
                self.pad_y = ((self.stride_y - 1) * H + self.kernel_height - self.stride_y) // 2
                self.pad_x = ((self.stride_x - 1) * W + self.kernel_width - self.stride_x) // 2
            else:
                self.pad_y, self.pad_x = 0, 0

        padded_image = self.pad_image(image)
        final_image = self.convolve(padded_image)

        return final_image

    @staticmethod
    def _validate_params(kernel_radius, stride, pad, pad_val, bias):
        """
        Validates input parameters.
        """
        if not (isinstance(kernel_radius, tuple) and len(kernel_radius) == 2 and
                all(isinstance(v, int) and v > 0 for v in kernel_radius)):
            raise ValueError(f"kernel_radius must be a tuple of two positive ints, got {kernel_radius}")

        if not (isinstance(stride, tuple) and len(stride) == 2 and
                all(isinstance(v, int) and v > 0 for v in stride)):
            raise ValueError(f"stride must be a tuple of two positive ints, got {stride}")

        if pad is not None:
            if not (isinstance(pad, tuple) and len(pad) == 2 and
                    all(isinstance(v, int) and v >= 0 for v in pad)):
                raise ValueError(f"pad must be a tuple of two non-negative ints, got {pad}")

        if not isinstance(pad_val, (int, float)):
            raise TypeError(f"pad_val must be a number, got {pad_val} instead")

        if not isinstance(bias, (int, float)):
            raise TypeError(f"bias must be a number, got {bias} instead")

