# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

d2net_path = Path(__file__).parent / "../third_party/d2net"
sys.path.append(str(d2net_path))
from third_party.d2net.lib.model_test import D2Net
from third_party.d2net.lib.pyramid import process_multiscale
from third_party.d2net.lib.utils import preprocess_image

from descriptors.Descriptor import Descriptor


class D2NetDescriptor(Descriptor):
    """Implementation of a D2Net descriptor using image-matching-toolbox.

    Args:
        device: Device to use for computation.
        config: Descriptor configuration.
    """

    def __init__(self, device: str, config: Dict[str, Any]):
        self.device = device
        use_cuda = True if self.device == "cuda" else False
        self.model = D2Net(config["model_path"], config["use_relu"], use_cuda)
        self.preprocessing = config["preprocessing"]
        self.scales = config["scales"]

    @torch.no_grad()
    def get_features(self, image: np.array) -> Tuple[np.array, np.array]:
        image = preprocess_image(image, preprocessing=self.preprocessing)
        keypoints, scores, descriptors = process_multiscale(
            torch.tensor(
                image[np.newaxis, :, :, :].astype(np.float32),
                device=self.device
            ),
            self.model,
            scales=self.scales
        )
        return keypoints[:, [1, 0]], descriptors

    def __str__(self) -> str:
        return "D2Net"
