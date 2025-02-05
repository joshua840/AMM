# Copyright (C) 2020-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from functools import partial
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .core import _CAM
import torch.nn.functional as F

__all__ = ["GradCAM", "LayerCAM", "GradCAMpp", "XGradCAM"]
# __all__ = ["GradCAM", "GradCAMpp", "LayerCAM", "SmoothGradCAMpp", "XGradCAM"]


class _GradCAM(_CAM):
    """Implements a gradient-based class activation map extractor.

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str = None,
    ) -> None:
        super().__init__(model, target_layer)
        # Ensure ReLU is applied before normalization
        self._relu = False


class GradCAM(_GradCAM):
    def _get_weights(self, activations, logits):
        grad = torch.autograd.grad(
            logits,
            activations,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad.mean(dim=(2, 3), keepdim=True)


class GradCAMpp(_GradCAM):
    def _get_weights(self, activations, logits):
        grad = torch.autograd.grad(
            logits,
            activations,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute pixel-wise weights (grad^2 / (2 * grad^2 + eps)) to avoid underflow
        grad_2 = grad.pow(2)
        grad_3 = grad_2 * grad
        denom = 2 * grad_2 + (grad_3 * activations).sum([2, 3], keepdim=True)

        # Avoid NaNs: Only compute for valid elements where grad_2 > 0
        valid_mask = grad_2 > 0
        alpha = grad_2
        # alpha[valid_mask] = alpha[valid_mask] / (denom[valid_mask] + 1e-8)

        # Compute final weights as the sum of weighted activations
        return F.adaptive_avg_pool2d(alpha * torch.relu(grad), 1)


class XGradCAM(_GradCAM):
    def _get_weights(self, activations, logits):
        grad = torch.autograd.grad(
            logits,
            activations,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
        )[0]
        return (grad * activations).sum([2, 3], keepdims=True) / activations.sum(
            [2, 3], keepdims=True
        ).add(1e-8)


class LayerCAM(_GradCAM):
    def _get_weights(self, activations, logits):
        grad = torch.autograd.grad(
            logits,
            activations,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
        )[0]

        return F.relu(grad)

    @staticmethod
    def _scale_cams(cams: List[Tensor], gamma: float = 2.0) -> List[Tensor]:
        # cf. Equation 9 in the paper
        return [torch.tanh(gamma * cam) for cam in cams]
