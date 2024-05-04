# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch

from descriptors.Descriptor import Descriptor
from third_party.superglue.models.superpoint import SuperPoint
from third_party.superglue.models.utils import frame2tensor


class SuperPointDescriptor(Descriptor):
    """Wrapper class for a SuperPoint descriptor.

    Args:
       device: Device to use for computation.
       config: Descriptor configuration.
    """

    def __init__(self, device: str, config: Dict[str, Any]):
        self.device = device
        self.descriptor = SuperPoint(config).eval().to(self.device)

    @torch.no_grad()
    def get_features(self, image: np.array) -> Tuple[np.array, np.array]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype("float32")
        gray = frame2tensor(gray, self.device)
        prediction = self.descriptor({'image': gray})
        keypoints = prediction['keypoints'][0]
        descriptors = prediction['descriptors'][0].permute(1, 0)
        return keypoints.cpu().numpy(), descriptors.cpu().numpy()

    def __str__(self) -> str:
        return "SuperPoint"
