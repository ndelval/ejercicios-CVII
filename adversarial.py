import torch
from torch import nn
from src.utils import visualize_perturbations


class CarliniWagnerL2:
    """
    White-box adversarial attack Carlini & Wagner (C&W) with L2 norm.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        targeted: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        model       : Model used for the attack.
        num_classes : Number of classes in the classification problem.
        targeted    : If True, performs a targeted attack. If False, untargeted.
        """
        self.model = model
        self.num_classes = num_classes
        self.targeted = targeted

    def _to_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts from pixel space [0, 1] to tanh space (-inf, +inf).

        Parameters
        ----------
        x : Image in pixel space [0, 1]. Dimensions: [channels, height, width].

        Returns
        -------
        w : Image in tanh space. Dimensions: [channels, height, width].
        """
        # TODO
        return torch.arctanh(2 * x - 1)

    def _from_tanh_space(self, w: torch.Tensor) -> torch.Tensor:
        """
        Converts from tanh space back to pixel space [0, 1].

        Parameters
        ----------
        w : Image in tanh space. Dimensions: [channels, height, width].

        Returns
        -------
        x : Image in pixel space [0, 1]. Dimensions: [channels, height, width].
        """
        # TODO
        return 0.5 * (torch.tanh(w) + 1)

    def _f_objective(
        self,
        adv_img: torch.Tensor,
        label: int,
        target_label: int | None = None,
        kappa: float = 0.0,
    ) -> torch.Tensor:
        """
        Computes the objective function f(x').

        Parameters
        ----------
        adv_img      : Adversarial image. Dimensions: [channels, height, width].
        label        : True label of the image.
        target_label : Target label for targeted attacks.
        kappa        : Confidence margin.

        Returns
        -------
        Objective function value (scalar tensor).
        """
        # TODO
        z_x = self.model(adv_img.unsqueeze(0))
        if not self.targeted:
            target_label = label
        idx = [i for i in range(self.num_classes) if i != target_label]
        z_t = z_x[..., target_label]
        z_i = torch.gather(z_x, 1, torch.tensor(idx).unsqueeze(0))
        if self.targeted:
            diff = z_i - z_t
        else:
            diff = -z_i + z_t
        max_diff = torch.amax(diff, dim=1)
        return torch.max(input=max_diff, other=torch.tensor(-kappa)).squeeze()

    def _compute_loss(
        self,
        w: torch.Tensor,
        original_img: torch.Tensor,
        label: int,
        c: float,
        target_label: int | None = None,
        kappa: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the total loss: L = ||x' - x||_2^2 + c * f(x')

        Parameters
        ----------
        w            : Optimization variable in tanh space.
        original_img : Original image. Dimensions: [channels, height, width].
        label        : True label of the image.
        c            : Constant for balancing the two objectives.
        target_label : Target label for targeted attacks.
        kappa        : Confidence margin.

        Returns
        -------
        loss    : Total loss value (scalar tensor).
        adv_img : Current adversarial image.
        """
        # TODO
        adv_img = self._from_tanh_space(w)
        f_x = self._f_objective(adv_img, label, target_label, kappa)
        norm = torch.norm(original_img - adv_img, p=2) ** 2
        return (norm + c * f_x, adv_img)

    def _check_success(
        self,
        adv_img: torch.Tensor,
        label: int,
        target_label: int | None = None,
    ) -> bool:
        """
        Checks if the attack was successful.

        Parameters
        ----------
        adv_img      : Adversarial image. Dimensions: [channels, height, width].
        label        : True label of the image.
        target_label : Target label for targeted attacks.

        Returns
        -------
        True if attack succeeded, False otherwise.
        """
        # TODO
        if not self.targeted:
            target_label = label
        output = self.model(adv_img.unsqueeze(0))
        value, idx = torch.max(output, dim=1)
        if self.targeted:
            return idx.item() == target_label
        else:
            return idx.item() != target_label

    def _optimize(
        self,
        img: torch.Tensor,
        label: int,
        c: float,
        max_iterations: int,
        learning_rate: float,
        target_label: int | None = None,
        kappa: float = 0.0,
    ) -> tuple[torch.Tensor, bool]:
        """
        Runs the optimization to find an adversarial example for a fixed c.

        Parameters
        ----------
        img            : Original image. Dimensions: [channels, height, width].
        label          : True label of the image.
        c              : Constant for the loss function.
        max_iterations : Number of optimization iterations.
        learning_rate  : Learning rate for the optimizer.
        target_label   : Target label for targeted attacks.
        kappa          : Confidence margin.

        Returns
        -------
        adv_img : Final adversarial image.
        success : Whether the attack succeeded.
        """
        # Clampeamos la imagen para evitar problemas con arctanh
        img_clamped = torch.clamp(img, 1e-6, 1 - 1e-6)

        w = self._to_tanh_space(img_clamped).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=learning_rate)
        best_adv = img.clone()
        best_l2 = float("inf")
        success = False
        for _ in range(max_iterations):
            optimizer.zero_grad()
            loss, current_adv = self._compute_loss(
                w, img, label, c, target_label, kappa
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if self._check_success(current_adv, label, target_label):
                    l2_dist = torch.norm(current_adv - img, p=2).item()
                    if l2_dist < best_l2:
                        best_l2 = l2_dist
                        best_adv = current_adv.clone()
                        success = True

        return best_adv, success

    def _binary_search_c(
        self,
        img: torch.Tensor,
        label: int,
        target_label: int | None,
        c_range: tuple[float, float],
        search_steps: int,
        max_iterations: int,
        learning_rate: float,
        kappa: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs binary search to find the optimal value of c.

        Parameters
        ----------
        img            : Original image. Dimensions: [channels, height, width].
        label          : True label of the image.
        target_label   : Target label for targeted attacks.
        c_range        : Initial range for c (c_lower, c_upper).
        search_steps   : Number of binary search iterations.
        max_iterations : Number of optimization iterations per c value.
        learning_rate  : Learning rate for the optimizer.
        kappa          : Confidence margin.

        Returns
        -------
        best_adv         : Best adversarial image found.
        best_perturbation : Best perturbation found.
        """
        c_lower, c_upper = c_range
        best_adv = img.clone()
        best_perturbation = torch.zeros_like(img)
        best_l2 = float("inf")

        for _ in range(search_steps):
            c = (c_lower + c_upper) / 2

            adv_img, success = self._optimize(
                img, label, c, max_iterations, learning_rate, target_label, kappa
            )

            if success:
                l2_dist = torch.norm(adv_img - img, p=2).item()
                if l2_dist < best_l2:
                    best_l2 = l2_dist
                    best_adv = adv_img.clone()
                    best_perturbation = adv_img - img
                # Si tuvo éxito, reducimos c para buscar menor perturbación
                c_upper = c
            else:
                # Si no tuvo éxito, aumentamos c
                c_lower = c

        return best_adv, best_perturbation

    def perturb_img(
        self,
        img: torch.Tensor,
        label: int,
        target_label: int | None = None,
        c: float = 1.0,
        kappa: float = 0.0,
        max_iterations: int = 1000,
        learning_rate: float = 0.01,
        binary_search_steps: int = 9,
        c_range: tuple[float, float] = (1e-3, 1e10),
        show: bool = True,
        **kwargs_visualize,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perturbs an image using the C&W L2 attack.

        Parameters
        ----------
        img                 : Original image. Dimensions: [channels, height, width].
        label               : True label of the image.
        target_label        : Target label for targeted attacks (None for untargeted).
        c                   : Initial constant for the loss function.
        kappa               : Confidence margin.
        max_iterations      : Number of optimization iterations.
        learning_rate       : Learning rate for Adam optimizer.
        binary_search_steps : Number of binary search steps for c.
        c_range             : Range for binary search of c.
        show                : Whether to visualize the results.

        Returns
        -------
        adv_img      : Adversarial image. Dimensions: [channels, height, width].
        perturbation : Perturbation applied. Dimensions: [channels, height, width].
        """
        if binary_search_steps > 0:
            # Usar búsqueda binaria para encontrar el mejor c
            adv_img, perturbation = self._binary_search_c(
                img,
                label,
                target_label,
                c_range,
                binary_search_steps,
                max_iterations,
                learning_rate,
                kappa,
            )
        else:
            # Usar el c dado directamente
            adv_img, success = self._optimize(
                img, label, c, max_iterations, learning_rate, target_label, kappa
            )
            perturbation = adv_img - img

        return adv_img, perturbation


import torch
from src.utils import visualize_perturbations


class ProjectedGradientDescent:
    """
    Skeleton implementation of the white-box Projected Gradient Descent (PGD)
    attack. PGD iteratively applies small gradient-based steps and projects the
    result back into an epsilon-ball around the original input.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        clamp: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """
        Parameters
        ----------
        model : Model to attack.
        loss  : Loss used to compute the gradients.
        clamp : Minimum and maximum values allowed for the image pixels.
        """

        self.model = model
        self.loss = loss
        self.clamp_min, self.clamp_max = clamp

    def _project(
        self,
        adv_img: torch.Tensor,
        original_img: torch.Tensor,
        epsilon: float,
        norm: str = "linf",
    ) -> torch.Tensor:
        """
        Projects the adversarial image back into the epsilon-ball centered at the
        original image and clamps it to the valid pixel range.

        TODO
        ----
        - Implement the projection for the L-infinity norm (and optionally L2).
        - Clip the perturbation so that ||adv_img - original_img||_p <= epsilon.
        - Clip the resulting image to [clamp_min, clamp_max].
        """
        delta = adv_img - original_img
        delta = torch.clamp(delta, -epsilon, epsilon)
        adv_proj = torch.clamp(original_img + delta, self.clamp_min, self.clamp_max)
        return adv_proj

    def _get_gradient(self, adv_img: torch.Tensor, label: int) -> torch.Tensor:
        """
        Computes the gradient of the loss with respect to the adversarial image.

        TODO
        ----
        - Zero previous gradients.
        - Enable gradient computation on the current adversarial image.
        - Forward pass through the model and compute the loss for the true label.
        - Backward pass to obtain the gradient.
        - Return the gradient tensor (for L-infinity PGD you will usually keep the
          sign of the gradient).
        """
        adv_leaf = adv_img.detach().clone()
        adv_leaf.requires_grad_(True)

        self.model.zero_grad(set_to_none=True)

        if adv_leaf.ndim == 3:
            adv_batch = adv_leaf.unsqueeze(0)
        else:
            adv_batch = adv_leaf

        output = self.model(adv_batch)
        target = torch.tensor([label], device=adv_batch.device)
        loss = self.loss(output, target)
        loss.backward()

        if adv_leaf.grad is None:
            raise RuntimeError("Gradient is None during PGD attack.")

        return adv_leaf.grad.detach()

    def perturb_img(
        self,
        img: torch.Tensor,
        label: int,
        epsilon: float = 0.03,
        step_size: float = 0.01,
        num_steps: int = 40,
        random_start: bool = True,
        norm: str = "linf",
        show: bool = True,
        **kwargs_visualize,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the PGD attack on a single image.

        TODO
        ----
        - Optionally start from a random point inside the epsilon-ball when
          random_start is True.
        - Iterate `num_steps` times:
            1. Compute the gradient on the current adversarial image.
            2. Take a step of size `step_size` in the gradient direction (for
               L-infinity you usually use the sign of the gradient).
            3. Project the result back into the epsilon-ball using `_project`.
        - Return the final adversarial image and the perturbation applied
          (adv_img - img).
        - When `show` is True, call `visualize_perturbations` with the final
          adversarial image, the original image, and any additional kwargs.
        """
        adv_img = img.detach().clone()

        if random_start:
            noise = torch.empty_like(img).uniform_(-epsilon, epsilon)
            adv_img = self._project(img + noise, img, epsilon, norm)

        for _ in range(num_steps):
            grad = self._get_gradient(adv_img, label)

            if norm == "linf":
                step = step_size * grad.sign()
            else:
                step = step_size * grad

            adv_img = adv_img + step
            adv_img = self._project(adv_img, img, epsilon, norm)

        perturbation = adv_img - img

        if show:
            visualize_perturbations(
                perturbed_img=adv_img.detach(),
                img=img.detach(),
                label=label,
                model=self.model,
                **kwargs_visualize,
            )

        return adv_img, perturbation
