import os
from typing import Literal

import numpy as np
from skimage import segmentation, color
from skimage.future import graph
from skimage import io
from skimage.feature import selective_search

ss = selective_search

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from utils import normalize_tensor, saliency_map


class SaliencyMaps(nn.Module):
    """Simple vanilla gradient saliency maps.

    Returns a normalized [batch, H, W] map.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:

        if x.dim() != 4:
            raise ValueError("Input tensor must be 4D.")

        x = x.clone().detach().requires_grad_(True)
        output = self.model(x)

        if target_class is None:
            score = output.max(1)[0].sum()
        else:
            score = output[
                range(x.shape[0]),
                torch.full(
                    (x.shape[0],),
                ),
                target_class,
            ].sum()
        score.backward()
        saliency = x.grad.abs().sum(dim=1)
        saliency_min = saliency.view(saliency.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
        saliency_max = saliency.view(saliency.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
        saliency_norm = (saliency - saliency_min) / (saliency_max - saliency_min)
        return saliency_norm


class InputXGradient(nn.Module):
    """Input * Gradient attribution. Returns normalized [batch, H, W]."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Input tensor must be 4D.")

        x = x.clone().detach().requires_grad_(True)
        output = self.model(x)

        if target_class is None:
            score = output.max(1)[0].sum()
        else:
            score = output[
                range(x.shape[0]),
                torch.full(
                    (x.shape[0],),
                ),
                target_class,
            ].sum()
        score.backward()
        saliency = (x * x.grad).abs().sum(dim=1)
        saliency_min = saliency.view(saliency.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
        saliency_max = saliency.view(saliency.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
        saliency_norm = (saliency - saliency_min) / (saliency_max - saliency_min)
        return saliency_norm


class GuidedBackpropagation(nn.Module):
    """Guided Backpropagation: modify ReLU backward pass so only positive gradients
    are propagated. Returns normalized [batch, H, W].
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.activation_maps: list[torch.Tensor] = []
        self.hooks = []
        # Ensure ReLUs are not inplace
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def _register_hooks(self) -> None:
        def forward_hook(module, inp, out):
            # store activations
            self.activation_maps.append(out.detach().clone())

        def backward_hook(module, grad_in, grad_out):
            if len(self.activation_maps) == 0:
                return grad_out
            activation = self.activation_maps.pop()
            if grad_out[0] is None:
                return grad_out
            guided = grad_out[0].detach().clone()
            mask = activation > 0
            guided = guided * mask.float()
            return (guided,)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                fh = module.register_forward_hook(forward_hook)
                bh = module.register_full_backward_hook(backward_hook)
                self.hooks.append(fh)
                self.hooks.append(bh)

    def _remove_hooks(self) -> None:
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()
        self.activation_maps.clear()

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        if not any(isinstance(m, nn.ReLU) for m in self.model.modules()):
            raise RuntimeError("No ReLU modules detected in the model")

        self._register_hooks()
        try:
            x = x.clone().detach().requires_grad_(True)
            output = self.model(x)
            if target_class is None:
                score = output.max(1)[0].sum()
            else:
                score = output[
                    range(x.shape[0]),
                    torch.full(
                        (x.shape[0],), target_class, dtype=torch.long, device=x.device
                    ),
                ].sum()
            score.backward()
            sal = x.grad.abs().sum(dim=1)
            return normalize_tensor(sal)
        finally:
            self._remove_hooks()


class DeconvNet(nn.Module):
    """DeconvNet style attribution (clamps backward gradient to positives)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.hooks = []
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def _register_hooks(self):
        def hook(module, grad_in, grad_out):
            if grad_out[0] is None:
                return grad_out
            return (torch.clamp(grad_out[0], min=0.0),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                h = module.register_full_backward_hook(hook)
                self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        self._register_hooks()
        try:
            x = x.clone().detach().requires_grad_(True)
            output = self.model(x)
            if target_class is None:
                score = output.max(1)[0].sum()
            else:
                score = output[
                    range(x.shape[0]),
                    torch.full(
                        (x.shape[0],), target_class, dtype=torch.long, device=x.device
                    ),
                ].sum()
            score.backward()
            sal = x.grad.abs().sum(dim=1)
            return normalize_tensor(sal)
        finally:
            self._remove_hooks()


class SmoothGradient(nn.Module):
    """SmoothGrad: average gradients over noisy samples."""

    def __init__(
        self, model: nn.Module, n_samples: int = 50, noise_level: float = 0.1
    ) -> None:
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.noise_level = noise_level

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        accumulated = torch.zeros_like(x, device=x.device)

        for _ in range(self.n_samples):
            noise = torch.rand_like(x) * self.noise_level
            x_noisy = (x + noise).clone().detach().requires_grad_(True)
            output = self.model(x_noisy)

            if target_class is None:
                score = output.max(1)[0].sum()
            else:
                score = output[
                    range(x.shape[0]), torch.full((x.shape[0],), target_class)
                ].sum()

            score.backward()
            accumulated += x_noisy.grad
            self.model.zero_grad()

        avg = accumulated / float(self.n_samples)
        saliency = avg.abs.sum(dim=1)

        saliency_min = saliency.view(saliency.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
        saliency_max = saliency.view(saliency.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
        saliency_norm = (saliency - saliency_min) / (saliency_max - saliency_min)
        return saliency_norm


class IntegratedGradients(nn.Module):
    """
    Computes Integrated Gradients for model interpretability.

    This class implements the Integrated Gradients method, which attributes the
    prediction of a model to its input features by integrating gradients along a
    straight path from a baseline to the input.

    More details can be found in: https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor of the class.

        Args:
            model: Model to explain.
        """

        super().__init__()

        self.model = model

    def _initialize_baseline(
        self,
        channels: int,
        height: int,
        width: int,
        baseline: Literal["zero", "random"],
    ) -> torch.Tensor:
        """
        Initializes the baseline tensor.

        Args:
            channels: Number of channels of the image.
            height: Height of the image.
            width: Width of the image.
            baseline: Type of baseline to use.

        Returns:
            Baseline tensor of shape (channels, height, width).

        Raises:
            ValueError: If the value of 'baseline' is not valid.
        """

        match baseline:
            case "zero":
                return torch.zeros((channels, height, width))
            case "random":
                return torch.randn((channels, height, width))
            case _:
                raise ValueError("Please introduce a correct value for the baseline.")

    def forward(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
        n_steps: int = 10,
        baseline: Literal["zero", "random"] = "zero",
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.
            n_steps: Number of steps for the integration path from baseline to input.
            baseline: Type of baseline to use.

        Returns:
            Explanation. Dimensions: [batch, height, width].
        """

        # TODO

        batch_size, channels, height, width = x.shape

        # Initialize baseline tensor and expand to batch size
        baseline_tensor = self._initialize_baseline(channels, height, width, baseline)
        baseline_tensor = baseline_tensor.to(x.device)
        baseline_tensor = baseline_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Create scaled inputs along the path from baseline to input
        scaled_inputs = []
        for i in range(n_steps + 1):
            alpha = float(i) / n_steps
            scaled_input = baseline_tensor + alpha * (x - baseline_tensor)
            scaled_inputs.append(scaled_input)

        # Stack all scaled inputs: (n_steps+1, batch, C, H, W)
        scaled_inputs = torch.stack(scaled_inputs, dim=0)

        # Reshape to process all at once: ((n_steps+1)*batch, C, H, W)
        scaled_inputs = scaled_inputs.view(-1, channels, height, width)

        # Get gradients for all scaled inputs
        grads = saliency_map(self.model, scaled_inputs, target_class)

        # Reshape gradients back: (n_steps+1, batch, H, W)
        grads = grads.view(n_steps + 1, batch_size, height, width)

        # Average gradients across integration steps
        avg_grads = grads.mean(dim=0)  # (batch, H, W)

        # Calculate integrated gradients
        # We need to sum over channels, so we compute per-channel and then sum
        diff = x - baseline_tensor  # (batch, C, H, W)

        # Expand avg_grads to match diff dimensions: (batch, C, H, W)
        avg_grads_expanded = avg_grads.unsqueeze(1).expand(-1, channels, -1, -1)

        # Element-wise multiplication and sum over channels
        integrated_grads = (diff * avg_grads_expanded).sum(dim=1)  # (batch, H, W)

        return normalize_tensor(integrated_grads)


class GradCAM(nn.Module):
    """
    Template implementation of Grad-CAM.

    Grad-CAM uses the gradients flowing into a target convolutional layer to
    produce a coarse localization map highlighting important regions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Constructor of the class.

        Args:
            model: Model to explain.
            target_layer: Convolutional layer to hook for activations and gradients.
        """

        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.activation: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.hooks: list[RemovableHandle] = []

    def _register_hooks(self) -> None:
        """
        Registers forward and backward hooks on the target layer to capture activations
        and gradients needed for Grad-CAM.

        TODO:
            - Register a forward hook to store the target layer output (activation).
            - Register a full backward hook to store gradients wrt the target layer.
            - Save handles in self.hooks for later removal.
        """

        def forward_hook(
            module: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor
        ) -> None:
            """
            Forward hooks. We will not use this, but if we wanted to extract the
            activations of each layer, for example to visualize them, we could use this.

            Args:
                module: Module of the hook.
                inp: Input tensor.
                out: Output tensor.

            Returns:
                Output tensor.
            """
            if module is self.target_layer:
                self.activation = out
                self.activation.requires_grad_(True)
            return out

        def backward_hook(
            module: nn.Module,
            grad_in: tuple[torch.Tensor, ...] | torch.Tensor,
            grad_out: tuple[torch.Tensor, ...] | torch.Tensor,
        ) -> tuple[torch.Tensor, ...] | torch.Tensor | None:
            """
            Backward hooks.

            Args:
                module: Module of the hook.
                gran_in: Input gradient.
                grad_out: Output gradient.

            Returns:
                Output gradient.
            """
            # Importante para guardarte el gradiente de alguna activacion intermedia y tal lo mejor es hacer esto
            grad_out = grad_out[0] if isinstance(grad_out, tuple) else grad_out
            if module is self.target_layer:
                self.gradients = grad_out
            return (grad_out,)

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                # Importante hacer el register sobre el module
                self.hooks.append(module.register_full_backward_hook(backward_hook))
                self.hooks.append(module.register_forward_hook(forward_hook))

    def _delete_hooks(self) -> None:
        """
        Deletes all registered hooks and clears cached activations/gradients.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].
            target_class: Class index for which the explanation is computed. If None,
                it uses the class with the highest score for each sample.

        Returns:
            Explanation. Dimensions: [batch, height, width].

        TODO:
            - Validate that target_layer belongs to the model.
            - Register hooks, run forward, pick target_class, run backward.
            - Compute channel weights from gradients, combine with activation.
            - Apply ReLU, upsample to input size, normalize with normalize_tensor.
            - Remove hooks before returning.
        """
        if not any(module is self.target_layer for module in self.model.modules()):
            raise RuntimeError("target_layer must belong to the provided model.")
        self._register_hooks()
        x.requires_grad_(True)
        self.model.zero_grad()
        out = self.model(x)
        if target_class is None:
            target_class = torch.argmax(out, dim=1).unsqueeze(-1)
        else:
            target_class = torch.full((out.size(0), 1), target_class)
        scores = torch.gather(out, 1, target_class).sum()
        scores.backward()

        alpha = torch.mean(self.gradients, dim=(2, 3))
        weighted_activations = alpha.unsqueeze(-1).unsqueeze(-1) * self.activation
        Mc = nn.ReLU()(torch.sum(weighted_activations, dim=1))
        self._delete_hooks()
        return Mc
