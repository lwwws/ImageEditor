import numpy as np
from ..base import BaseFilter
from .box import Box

"""
ChatGPT Usage:
The idea of creating such filter, using kron to upsample the image
"""
class Retro(BaseFilter):
    """
    Simulates pixelation by reducing resolution in blocks and expanding back.

    Parameters:
        block_size (int): Size of each pixel block in output. Must be a power of two.
    """

    def __init__(self, block_size=2, **kwargs):
        if block_size & (block_size - 1) != 0 or block_size <= 0:
            raise ValueError(f"block_size must be a power of 2, got {block_size}")

        self.block_size = block_size
        kernel_size = block_size * 2 + 1

        # Default to Box blur with block-size kernel if no blur filter is passed
        self.blur = Box(
            height=kernel_size,
            width=kernel_size,
            stride=(block_size, block_size),
            keep_dims=False,
            **kwargs
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applies block averaging and expands result back to original size.
        """
        out = self.blur.apply(image)
        upsampled = np.kron(out, np.ones((self.block_size, self.block_size, 1)))
        return upsampled
