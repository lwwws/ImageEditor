import numpy as np
from ..base import StaticFilter

"""
ChatGPT Usage:
Creating precomputed direction kernels
"""

DIRECTION_KERNELS = {
        'northeast': np.array([[-2, -1,  0],
                               [-1,  1,  1],
                               [ 0,  1,  2]]),

        'southwest': np.array([[ 2,  1,  0],
                               [ 1, -1, -1],
                               [ 0, -1, -2]]),

        'northwest': np.array([[ 0,  1,  2],
                               [-1,  1,  1],
                               [-2, -1,  0]]),

        'southeast': np.array([[ 0, -1, -2],
                               [ 1,  1, -1],
                               [ 2,  1,  0]]),

        'top': np.array([[ 1,  1,  1],
                         [ 0,  0,  0],
                         [-1, -1, -1]]),

        'left': np.array([[ 1,  0, -1],
                          [ 1,  0, -1],
                          [ 1,  0, -1]]),
    }

class Emboss(StaticFilter):
    """
    Applies a directional emboss effect to simulate lighting and depth.

    The emboss effect highlights edges in a given direction, creating a stylized
    3D-relief look. Can be used for texture stylization or pseudo-depth.

    Parameters:
        strength (float): Multiplier for intensity of emboss.
        direction (str): Light source direction. Options: 'northeast', 'southwest',
                         'northwest', 'southeast', 'top', 'left'.
    """

    def __init__(self, strength=1.0, direction='northeast', **kwargs):
        if direction not in DIRECTION_KERNELS:
            raise ValueError(f"Unsupported direction: {direction}")

        base_kernel = DIRECTION_KERNELS[direction] * strength
        kernel = base_kernel[:, :, np.newaxis]
        super().__init__(kernels=kernel, **kwargs)
