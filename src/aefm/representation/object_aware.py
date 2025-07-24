from typing import Callable, Dict, Optional

import schnetpack.nn as snn
import torch
import torch.nn.functional as F
from schnetpack.representation.painn import PaiNN
from torch import nn
from torch_scatter import scatter_mean

from aefm import properties

__all__ = ["PaiNNObjectAware", "FragmentMixing"]


class FragmentMixing(nn.Module):

    def __init__(self, n_atom_basis: int, activation: Callable):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis

        self.interfragment_edge_net = nn.Sequential(
            snn.Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, n_atom_basis, activation=activation),
        )

        self.interfragment_context_net = nn.Sequential(
            snn.Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, n_atom_basis, activation=None),
        )

    def forward(
        self,
        q: torch.Tensor,
        idx_i_fragment: torch.Tensor,
        idx_j_fragment: torch.Tensor,
    ):

        source = q[idx_i_fragment]
        target = q[idx_j_fragment]
        mij = torch.cat([source, target], dim=-1)
        mij = self.interfragment_edge_net(mij)

        # Aggregate the messages (sum explodes because of many neighbors)
        agg = scatter_mean(
            mij,
            idx_i_fragment,
            dim=0,
            dim_size=q.shape[0],
        )

        # Update the atom embeddings
        agg = torch.cat([q, agg], dim=-1)
        q = self.interfragment_context_net(agg)

        return q


class PaiNNObjectAware(PaiNN):
    """PaiNN - object aware"""

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        activation: Optional[Callable] = F.silu,
        shared_interactions: bool = False,
        p_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            activation=activation,
            **kwargs,
        )

        # object-aware scalar message passing between fragments
        self.dropout = nn.Dropout(p=p_dropout)
        self.fragment_mixing = snn.replicate_module(
            lambda: FragmentMixing(
                n_atom_basis=self.n_atom_basis, activation=activation
            ),
            self.n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs: SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        n_atoms = atomic_numbers.shape[0]

        # get neighbor list for interfragment interactions
        idx_i_fragment = inputs[properties.idx_i_fragment]
        idx_j_fragment = inputs[properties.idx_j_fragment]

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        # compute initial embeddings
        q = self.embedding(atomic_numbers)
        for embedding in self.electronic_embeddings:
            q = q + embedding(q, inputs)
        q = q.unsqueeze(1)

        # compute interaction blocks and update atomic embeddings
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)
        for i, (interaction, mixing, fragment_mixing) in enumerate(
            zip(self.interactions, self.mixing, self.fragment_mixing)
        ):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
            q = self.dropout(q)
            q = fragment_mixing(q, idx_i_fragment, idx_j_fragment)

        q = q.squeeze(1)

        # collect results
        inputs["scalar_representation"] = q
        inputs["vector_representation"] = mu

        return inputs
