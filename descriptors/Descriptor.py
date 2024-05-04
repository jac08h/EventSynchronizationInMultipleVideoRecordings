# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Descriptor(ABC):
    """
    Base class for an image descriptor.
    """

    @abstractmethod
    def get_features(self, image: np.array) -> Tuple[np.array, np.array]:
        """Detect and describe features on an image.

        Args:
            image: RGB image in (H, W, 3) shape.

        Returns:
            Tuple[np.array, np.array]: Keypoints in (N, 2) shape and descriptors in (N, D) shape, where
                N is number of detected keypoints, D is a descriptor's dimension.
        """
        pass
