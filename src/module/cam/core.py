# Copyright (C) 2020-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import logging
from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


__all__ = ["_CAM"]


class _CAM:
    """Implements a class activation map extractor

    Args:
        model: input model
        target_layer: we only accept single target layer
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str = None,
    ) -> None:
        self.target_module = dict(model.named_modules())[target_layer]
        self.target_module.register_forward_hook(self._hook)
        self._relu = False

    @staticmethod
    def _hook(module, input, output):
        module.output = output

    @staticmethod
    def _normalize(cams: Tensor, eps: float = 1e-8) -> Tensor:
        max_val = cams.abs().amax(dim=(-2, -1), keepdim=True)
        cams = cams / (max_val + eps)
        return cams

    @abstractmethod
    def _get_weights(self, activations, logits):
        raise NotImplementedError

    def compute_cams(
        self,
        logits: Tensor,
        normalized: bool = True,
    ):
        # Get map weight & unsqueeze it

        activations = self.target_module.output
        weights = self._get_weights(activations, logits)

        cam = torch.nansum(weights * activations, dim=1)

        if self._relu:
            cam = F.relu(cam, inplace=False)

        # Normalize the CAM
        if normalized:
            cam = self._normalize(cam)
        return cam
