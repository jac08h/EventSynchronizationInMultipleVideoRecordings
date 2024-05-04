# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from descriptors.Descriptor import Descriptor

sys.path.append(str(Path(__file__).parent / '../third_party/r2d2'))
from third_party.r2d2.extract import NonMaxSuppression, extract_multiscale
from third_party.r2d2.nets.patchnet import *
from third_party.r2d2.tools.dataloader import norm_RGB


class R2D2Descriptor(Descriptor):
    """Wrapper class for a R2D2 descriptor.

    Args:
        device: Device to use for computation.
        config: Descriptor configuration.
    """

    def __init__(self, device: str, config: Dict[str, Any]):
        self.device = device
        model = torch.load(config["model_path"])
        self.descriptor = eval(model['net']).to(self.device).eval()
        self.descriptor.load_state_dict({k.replace('module.', ''): v for k, v in model['state_dict'].items()})
        self.detector = NonMaxSuppression(rel_thr=config["reliability_thr"], rep_thr=config["repeatability_thr"])
        self.config = config

    @torch.no_grad()
    def get_features(self, image: np.array) -> Tuple[np.array, np.array]:
        image = norm_RGB(image)[None].to(self.device)
        xys, desc, scores = extract_multiscale(self.descriptor,
                                               image,
                                               self.detector,
                                               min_scale=self.config["min_scale"],
                                               max_scale=self.config["max_scale"],
                                               min_size=0,
                                               max_size=9999
                                               )
        idxs = scores.argsort()[-self.config["top_k"] or None:]
        keypoints = xys[idxs][:, :2]
        descriptors = desc[idxs]
        return keypoints.cpu().numpy(), descriptors.cpu().numpy()

    def __str__(self) -> str:
        return "R2D2"
