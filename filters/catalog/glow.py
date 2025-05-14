from ..base import DynamicFilter
import numpy as np

class Glow(DynamicFilter):
    def __init__(self, low_thresh=40, high_thresh=80, glow_boost=2.0, **kwargs):
        """
        Initializes thresholds for detecting dim glows.

        Parameters:
            low_thresh (int): Minimum brightness to consider as glow.
            high_thresh (int): Upper limit to avoid boosting already bright areas.
            glow_boost (float): Boost factor for glow regions.
        """

        super().__init__(**kwargs)
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.glow_boost = glow_boost

    def apply_region(self, region: np.ndarray, y: int, x: int) -> np.ndarray:
        """
        Amplifies mid-brightness (glow) areas to reveal hidden content.
        """
        center_pixel = region[self.radius_y, self.radius_x]
        brightness = np.mean(region)

        if self.low_thresh < brightness < self.high_thresh:
            return center_pixel * self.glow_boost
        else:
            return center_pixel
