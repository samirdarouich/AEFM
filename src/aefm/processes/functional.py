from typing import Optional, Tuple

import torch

from aefm.utils import batch_center_systems


def _check_shapes(
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Checks and fixes the shapes t.

    Args:
        x: input tensor.
        t: current time steps.

    """
    if len(x.shape) < len(t.shape):
        x = x.unsqueeze(0).repeat_interleave(t.shape[0], 0)

    while len(x.shape) > len(t.shape):
        t = t.unsqueeze(-1)

    return x, t


def sample_noise(
    shape: Tuple,
    invariant: bool,
    idx_m: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Sample Gaussian noise based on input shape.
    Project to the zero center of geometry if invariant.

    Args:
        shape: shape of the noise.
        invariant: if True, apply invariance constraint.
        idx_m: same as ``proporties.idx_m`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
        device: torch device to store the noise tensor.
        n_atoms: number of atoms per system.
        dtype: data type to use for computation accuracy.
    """
    # sample noise
    noise = torch.randn(shape, device=device, dtype=dtype)

    # The invariance trick: project noise to the zero center of geometry.
    if invariant:
        # system-wise center of geometry
        if idx_m is not None:
            noise = batch_center_systems(noise, idx_m, dim=-2)  # type: ignore

        # global center of geometry if one system passed.
        else:
            noise -= noise.mean(-2).unsqueeze(-2)

    return noise


def sample_noise_like(
    x: torch.Tensor,
    invariant: bool,
    idx_m: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Sample Gaussian noise based on input x.

    Args:
        x: input tensor, e.g. to infer shape.
        invariant: if True, apply invariance constraint.
        idx_m: same as ``proporties.idx_m`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
    """
    return sample_noise(x.shape, invariant, idx_m, device=x.device, dtype=x.dtype)


def sample_isotropic_Gaussian(
    mean: torch.Tensor,
    std: torch.Tensor,
    invariant: bool,
    idx_m: Optional[torch.Tensor],
    noise: Optional[torch.Tensor] = None,
):
    """
    Use the reparametrization trick to Sample from iso Gaussian distribution
    with given mean and std.

    Args:
        mean: mean of the Gaussian distribution.
        std: standard deviation of the Gaussian distribution.
        invariant: if True, the noise is projected to the zero center of geometry.
                The mean is computed over the -2 dimension.
        idx_m: same as ``proporties.idx_m`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
        noise: the Gaussian noise. If None, a new noise is sampled.
                Otherwise, ``idx_m`` and ``invariant`` are ignored.
    """
    # sample noise if not given.
    if noise is None:
        noise = sample_noise_like(mean, invariant, idx_m)

    # sample using the Gaussian reparametrization trick.
    sample = mean + std * noise

    return sample, noise
