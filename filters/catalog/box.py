import numpy as np
from ..base import StaticFilter


class Box(StaticFilter):
    def __init__(self, alpha: float = None, width: int = 3, height: int = 3, keep_dims=True, **kwargs):
        """
            Box blur filter using a uniform kernel.

            Parameters:
                alpha (float): Optional. Value in (0, 1] to scale blur radius.
                width, height (int): Optional. Exact kernel width and height.
                keep_dims (bool): Whether to pad image to preserve original size.
                **kwargs: Passed to StaticFilter/ConvFilter (e.g. pad_val, bias, ...).
        """

        base_radius_y, base_radius_x = (10, 10)

        if alpha is not None:
            if not isinstance(alpha, (int, float)) or not (0 < alpha <= 1):
                raise ValueError(f"Strength must be in (0, 1], got {alpha}")
            radius_y = max(1, int(round(base_radius_y * alpha)))
            radius_x = max(1, int(round(base_radius_x * alpha)))
        else:
            if not (1 <= height <= 2 * base_radius_x + 1 and 1 <= width <= 2 * base_radius_y + 1):
                raise ValueError(
                    f"Width, height must be odd and in range [1, {2 * base_radius_y + 1}], [1, {2 * base_radius_x + 1}] respectively"
                )
            if height % 2 == 0 or width % 2 == 0:
                raise ValueError(f"Kernel size must be odd, got ({width}, {height})")

            radius_y = width // 2
            radius_x = height // 2

        kernel_height = 2 * radius_y + 1
        kernel_width = 2 * radius_x + 1

        value = 1.0 / (kernel_width * kernel_height)

        kernel = np.full((kernel_height, kernel_width), value, dtype=np.float32)
        kernel = kernel[:, :, np.newaxis]

        super().__init__(kernels=kernel, keep_dims=keep_dims, **kwargs)
