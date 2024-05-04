# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from typing import Optional, Tuple

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
import torch.nn as nn
from kornia_moons.feature import *

from descriptors.Descriptor import Descriptor


class KorniaDescriptor(Descriptor):
    """Class for descriptors from Kornia. It detects keypoints using SIFT from OpenCV, extracts LAFs from the
        detected keypoints, optionally improves the LAFs using AffNet, and then uses the kornia descriptor to
        describe the LAFs.
        Based on https://github.com/kornia/kornia-examples/blob/75908eb9dd2de043562162ad2d639dc9c298fca7/MKD_TFeat_descriptors_in_kornia.ipynb

        Args:
            descriptor: Initialized Kornia descriptor.
            use_affnet: Use AffNet to estimate affine shape of patches detected by SIFT.
            device: Device to use for computation.
            max_sift_features: Number of best features to retain from SIFT detection. If None, retain all features.
    """

    def __init__(self, descriptor: nn.Module, use_affnet: bool, device: str, max_sift_features: Optional[int]):
        self.descriptor = descriptor.to(device)
        self.descriptor.eval()
        self.use_affnet = use_affnet
        self.device = device
        self.sift = cv2.SIFT_create(nfeatures=0 if max_sift_features is None else max_sift_features)
        if self.use_affnet:
            self.affine = KF.LAFAffNetShapeEstimator(True).to(device)
            self.orienter = KF.LAFOrienter(self.descriptor.patch_size, angle_detector=KF.OriNet(True)).to(device)
            self.orienter.eval()
            self.affine.eval()

    def get_local_descriptors(self, image: np.array, cv_sift_kpts: Tuple[cv2.KeyPoint, ...]) -> np.ndarray:
        """Describe frame features using detected keypoints.

        Args:
            image: RGB image in (H, W, 3) shape.
            cv_sift_kpts: Detected keypoints.

        Returns:
            np.ndarray: Descriptors in (N, D) shape where N is number of detected keypoints
                and D is a descriptor's dimension.
        """
        timg = K.color.rgb_to_grayscale(K.image_to_tensor(image, False).to(self.device).float()) / 255.
        lafs = laf_from_opencv_SIFT_kpts(cv_sift_kpts).to(self.device)
        if self.use_affnet:
            lafs = self.orienter(self.affine(lafs, timg), timg)
        patches = KF.extract_patches_from_pyramid(timg, lafs, self.descriptor.patch_size)
        B, N, CH, H, W = patches.size()
        descs = self.descriptor(patches.view(B * N, CH, H, W)).view(B * N, -1)
        return descs.detach().cpu().numpy()

    @torch.no_grad()
    def get_features(self, frame: np.array) -> Tuple[np.array, np.array]:
        cv_keypoints = self.sift.detect(frame, None)
        descriptors = self.get_local_descriptors(frame, cv_keypoints)
        keypoints = np.array([kp.pt for kp in cv_keypoints])
        return keypoints, descriptors

    def __str__(self):
        return str(self.descriptor)[:str(self.descriptor).index("(")]
