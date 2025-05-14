import numpy as np
from ..base import StaticFilter

"""
ChatGPT Usage:
Creating precomputed sobel kernels
"""

SOBEL_KERNELS = {
    3: (
        np.array([  # Kx
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32),
        np.array([  # Ky
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)
    ),
    5: (
        np.array([  # Kx (horizontal gradient)
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
            [-4, -2, 0, 2, 4],
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2]
        ], dtype=np.float32),
        np.array([  # Ky (vertical gradient)
            [-2, -2, -4, -2, -2],
            [-1, -1, -2, -1, -1],
            [ 0,  0,  0,  0,  0],
            [ 1,  1,  2,  1,  1],
            [ 2,  2,  4,  2,  2]
        ], dtype=np.float32)
    )
}

class Sobel(StaticFilter):
    def __init__(self, size=3, keep_dims=True, **kwargs):
        """
        Sobel Filter.
        Applies Sobel edge detection using horizontal and vertical derivative filters.

        Parameters:
            size (int): Kernel size (3 or 5 supported).
            keep_dims (bool): If True, pads to preserve dimensions.
        """

        self.Kx, self.Ky = self.get_sobel_kernels(size)
        radius = self.Kx.shape[0], self.Kx.shape[0]

        super().__init__(kernels=[self.Kx, self.Ky], keep_dims=keep_dims, **kwargs)

    @staticmethod
    def get_sobel_kernels(size: int):
        """
        Retrieve pre computed Sobel kernels.
        """
        try:
            Kx, Ky = SOBEL_KERNELS[size]
            return Kx[:, :, np.newaxis], Ky[:, :, np.newaxis]
        except KeyError:
            raise ValueError(f"No Sobel kernels defined for size {size}")

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Apply horizontal and vertical filters separately
        Gx = self._apply_single_filter(image, self.kernels[0])
        Gy = self._apply_single_filter(image, self.kernels[1])

        # Compute gradient magnitude
        G = np.sqrt(Gx.astype(np.float32) ** 2 + Gy.astype(np.float32) ** 2)
        return np.clip(G, 0, 255)


