"""
DynamicFilter Module
-----------------
Abstract base class for filters that apply custom logic per image region.

Unlike StaticFilter which uses fixed kernels, DynamicFilter allows you to compute
the output for each pixel dynamically, based on the local region.

Runs slower than StaticFilters Module.

Inherits padding, stride, and clipping logic from ConvFilter.

Subclasses must implement 'apply_region()' to define how to transform each region.

Author: lwwws
"""

from abc import abstractmethod
import numpy as np
from .conv_filter import ConvFilter
class DynamicFilter(ConvFilter):
    """
    A convolutional filter base that computes output per region using dynamic logic.

    This is useful for non-linear filters, adaptive kernels, or filters where the
    convolution result depends on local image properties.
    """

    def __init__(self, **kwargs):
        """
        Inherits convolution and padding parameters from ConvFilter.
        """
        super().__init__(**kwargs)

    @abstractmethod
    def apply_region(self, region: np.ndarray, y: int, x: int) -> np.ndarray:
        """
        Compute output value for a given region.

        Subclasses must override this method to define how a local patch of pixels
        contributes to the output.

        Parameters:
            region (np.ndarray): Image patch of shape (H, W, C)
            y (int): Output pixel row index
            x (int): Output pixel column index

        Returns:
            np.ndarray: Pixel value to write at output[y, x, :]
        """
        pass

    def compute_convolution(self, region: np.ndarray, kernel: np.ndarray) -> float:
        """
        Utility function to compute standard convolution on a region.

        Parameters:
            region (np.ndarray): Patch of the image
            kernel (np.ndarray): Convolution kernel (same shape as region)

        Returns:
            float: Convolution result
        """

        return np.sum(region * kernel, axis=(0, 1)) + self.bias

    def convolve(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the region-wise logic over the entire padded image.

        Parameters:
            image (np.ndarray): Padded input image (H, W, C)

        Returns:
            np.ndarray: Output image after region-wise filtering
        """

        # Output image dimensions
        n_channels = image.shape[-1]
        out_height = (image.shape[0] - self.kernel_height) // self.stride_y + 1
        out_width = (image.shape[1] - self.kernel_width) // self.stride_x + 1

        out = np.zeros((out_height, out_width, n_channels))

        for y in range(out_height):
            for x in range(out_width):
                center_y = y * self.stride_y + self.radius_y
                center_x = x * self.stride_x + self.radius_x

                # Extract region around the center pixel
                region = image[
                         center_y - self.radius_y: center_y + self.radius_y + 1,
                         center_x - self.radius_x: center_x + self.radius_x + 1,
                         :
                         ]

                out[y, x, :] = np.clip(self.apply_region(region, y, x), 0, 255)

        return out
