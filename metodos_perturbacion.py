import torch
import torch.nn.functional as F
from torch import nn

from utils import normalize_tensor


class SHAP(nn.Module):
    """Very small approximate SHAP-like estimator using random linear interpolations.

    Returns a normalized [batch, H, W] map.
    """

    def __init__(
        self,
        model: nn.Module,
        baseline: torch.Tensor | None = None,
        n_samples: int = 50,
    ) -> None:
        super().__init__()
        self.model = model
        self.baseline = baseline
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        pass


class Occlusion(nn.Module):
    """
    Computes Occlusion for model interpretability.

    This class provides methods to generate feature or pixel-level importance
    maps by systematically occluding parts of the input and measuring the
    impact on the model's output.

    More details can be found in: https://arxiv.org/abs/1311.2901
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor of the class.

        Args:
            model: Model to explain.
        """

        super().__init__()

        self.model = model

    def _get_target_outputs(
        self, original_output: torch.Tensor, target_class: int | None
    ):
        """
        Gets the output values and target indices for the specified class.

        Args:
            original_output: The model's output tensor.
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.

        Returns:
            Output values for the target class and target class indices as a tensor.

        Raises:
            ValueError: If target_class is not None or an integer.
        """
        if target_class is None:
            target_indices = original_output.argmax(dim=1).unsqueeze(1)
            out_vals = torch.gather(original_output, 1, target_indices).squeeze(1)
            return out_vals, target_indices
        elif isinstance(target_class, int):
            target_indices = torch.full(
                (original_output.size(0), 1),
                target_class,
                device=original_output.device,
            )
            out_vals = torch.gather(original_output, 1, target_indices).squeeze(1)
            return out_vals, target_indices
        else:
            raise ValueError("target_class debe ser None o un entero")

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
        mask_size: int | tuple[int, int] | None = None,
        stride: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, channels, height, width].
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.
            mask_size: Shape of the occlusion mask. If int, creates a square mask. If
                tuple, uses (width, height). If None, it uses a mask ten times smaller
                than the width and height.
            stride: Stride for moving the occlusion mask. If None, it uses half the mask
                size.

        Returns:
            Explanation. Dimensions: [batch, height, width].
        """
        batch_size, _, height, width = x.shape

        if mask_size is None:
            mask_h = max(1, height // 10)
            mask_w = max(1, width // 10)
        elif isinstance(mask_size, int):
            mask_h = mask_w = max(1, mask_size)
        else:
            mask_w, mask_h = mask_size
            mask_h = max(1, mask_h)
            mask_w = max(1, mask_w)

        mask_h = min(mask_h, height)
        mask_w = min(mask_w, width)

        if stride is None:
            stride_h = max(1, mask_h // 2)
            stride_w = max(1, mask_w // 2)
        else:
            stride_h = stride_w = max(1, stride)

        original_output = self.model(x)
        out_vals, target_indices = self._get_target_outputs(
            original_output, target_class
        )

        saliency_maps = torch.zeros((batch_size, height, width), device=x.device)
        counts = torch.zeros((batch_size, height, width), device=x.device)

        for h_start in range(0, height - mask_h + 1, stride_h):
            for w_start in range(0, width - mask_w + 1, stride_w):
                masked_x = x.clone()
                masked_x[
                    :, :, h_start : h_start + mask_h, w_start : w_start + mask_w
                ] = 0

                output_occluded = self.model(masked_x)
                occluded_vals = torch.gather(output_occluded, 1, target_indices)

                diff = out_vals - occluded_vals.squeeze(1)

                for i in range(batch_size):
                    saliency_maps[
                        i, h_start : h_start + mask_h, w_start : w_start + mask_w
                    ] += diff[i]
                counts[:, h_start : h_start + mask_h, w_start : w_start + mask_w] += 1

        saliency_maps = saliency_maps / (counts + 1e-8)
        return normalize_tensor(saliency_maps)


class RISE(nn.Module):
    """
    Template implementation of Randomized Input Sampling for Explanation (RISE).

    RISE estimates pixel importances by averaging model scores over randomly
    masked versions of the input. The core idea is:
    - Sample many random binary masks.
    - Resize them to the input resolution.
    - Multiply each mask with the input, run the model and keep the score for the
      target class.
    - Weight and sum the masks with those scores to obtain a saliency map.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor of the class.

        Args:
            model: Model to explain.
        """

        super().__init__()
        self.model = model

    def _generate_masks(
        self,
        n_masks: int,
        input_size: tuple[int, int],
        mask_size: tuple[int, int],
        p: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generates random binary masks and upsamples them to match the input size.

        Args:
            n_masks: Number of masks to sample.
            input_size: Tuple with (height, width) of the input image.
            mask_size: Size of the low-resolution mask before upsampling.
            p: Probability of a 1 in the Bernoulli masks.
            device: Device where the masks should live.

        Returns:
            Tensor with shape [n_masks, 1, height, width] containing the masks.

        TODO:
            1. Sample Bernoulli masks of shape [n_masks, 1, mask_h, mask_w].
            2. Upsample them to [n_masks, 1, height, width] (nearest/bilinear).
            3. Optionally jitter masks before interpolation (standard RISE trick).
        """
        mask_h, mask_w = mask_size
        height, width = input_size

        mask = torch.full((n_masks, 1, mask_h, mask_w), p)
        mask = torch.bernoulli(mask)
        mask_upsamled = torch.nn.functional.interpolate(mask, size=(height, width))
        return mask_upsamled

    def _collect_scores(
        self, outputs: torch.Tensor, target_class: int | None
    ) -> torch.Tensor:
        """
        Selects the scores of the target class for each masked input.

        Args:
            outputs: Model outputs for each masked input. Shape: [batch, classes].
            target_class: Class index to explain. If None, use the argmax per sample.

        Returns:
            Scores tensor with shape [batch].

        TODO:
            - When target_class is None, pick the top-1 class per masked sample.
            - Otherwise gather the score of target_class for every sample.
        """
        if target_class is None:
            target_class = torch.argmax(outputs, dim=1).unsqueeze(-1)
        else:
            target_class = torch.full((outputs.size(0), 1), target_class)
        scores = torch.gather(outputs, 1, target_class)
        return scores

    def forward(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
        n_masks: int = 1000,
        p: float = 0.5,
        mask_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].
            target_class: Class index for which the explanation is computed. If None,
                it uses the class with the highest score for each sample.
            n_masks: Number of random masks to sample.
            p: Probability of a 1 in each Bernoulli mask.
            mask_size: Size of the low-res mask before upsampling. If None, a default
                proportional to the input size should be used.

        Returns:
            Explanation. Dimensions: [batch, height, width].

        TODO:
            - Validate the inputs (p in (0, 1], n_masks > 0, mask_size not too small).
            - Decide a default mask_size when None (e.g. input_size // 7).
            - Generate masks and expand them to match the batch.
            - Apply masks to the input, run the model and collect scores.
            - Aggregate the weighted masks into a saliency map (normalize by p * n_masks).
            - Normalize the final saliency with normalize_tensor and return it.
        """
        batch, channels, height, width = x.size()
        if n_masks <= 0:
            raise ValueError("n_masks must be greater than 0.")
        if not (0 < p <= 1):
            raise ValueError("p must be in the interval (0, 1].")
        if mask_size is not None:
            mask_h, mask_w = mask_size
            if mask_h <= 0 or mask_w <= 0:
                raise ValueError("mask_size dimensions must be positive.")
        if mask_size is None:
            mask_size = (max(1, height // 7), max(1, width // 7))

        rise_values = torch.zeros_like(x)
        seen_pixels = torch.zeros_like(x)

        mask = self._generate_masks(
            n_masks, (height, width), mask_size, p, device="cpu"
        )
        masked_x = (x.unsqueeze(0) * mask.unsqueeze(1)).view(
            -1, channels, height, width
        )
        output = self.model(masked_x)
        scores = (
            self._collect_scores(outputs=output, target_class=target_class)
            .view(n_masks, batch, 1)
            .sum(0)
        )
        seen_pixels += mask.sum(0).unsqueeze(0)
        rise_values += scores.unsqueeze(-1).unsqueeze(-1)
        output = (rise_values / (seen_pixels + 1e-8)).sum(dim=1)
        return normalize_tensor(output)
