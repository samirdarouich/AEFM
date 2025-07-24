import logging

import torch
from torch_scatter import scatter_mean

log = logging.getLogger(__name__)


def batch_center_systems(systems: torch.Tensor, idx_m: torch.Tensor, dim: int = 0):
    """
    center batch of systems moleculewise to have zero center of geometry

    Args:
        systems (torch.tensor): batch of systems (molecules)
        idx_m (torch.tensor): the system id for each atom in the batch
        dim (int): dimension to scatter over
    """
    mean = scatter_mean(systems, idx_m, dim=dim)

    # broadcast mean to the same shape as systems
    mean = mean.movedim(dim, 0)[idx_m].movedim(0, dim)

    return systems - mean


def _check_times(times, t_0, timesteps):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= timesteps, (t, timesteps)


def get_repaint_schedule(timesteps, jump_length, resamplings):
    jumps = {}
    for j in range(0, timesteps - jump_length, jump_length):
        jumps[j] = resamplings - 1

    t = timesteps
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, timesteps)

    return ts
