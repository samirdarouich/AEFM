from typing import Dict, Optional, Tuple

import torch
from schnetpack.representation import PaiNN
from torch import nn

from aefm import properties

__all__ = ["PaiNNCombined", "PaiNNRPEmb"]


class PaiNNCombined(PaiNN):
    """PaiNN - combined"""

    def __init__(
        self,
        own_embedding_rp: bool = False,
        combine_mode: str = "diff",
        p_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert combine_mode in [
            "diff",
            "sum",
            "mean",
            "mlp",
        ]
        self.combine_mode = combine_mode
        if self.combine_mode == "mlp":
            self.combine_mlp = nn.Sequential(
                nn.Linear(2 * self.n_atom_basis, self.n_atom_basis),
                nn.SiLU(),
                nn.Linear(self.n_atom_basis, self.n_atom_basis),
            )
        self.rp_embedding = (
            nn.Embedding(100, self.n_atom_basis) if own_embedding_rp else None
        )
        self.project_scalar_down = nn.Linear(
            self.n_atom_basis * 2, self.n_atom_basis, bias=False
        )
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        atomic_numbers = inputs[properties.Z]
        coord_reactants = inputs[properties.reactant_coords]
        coord_products = inputs[properties.product_coords]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]

        # Get distances
        Rij_reactant = coord_reactants[idx_j] - coord_reactants[idx_i]
        Rij_product = coord_products[idx_j] - coord_products[idx_i]
        Rij = inputs[properties.Rij]

        # Batch to one graph
        offset = idx_i.max() + 1
        atomic_numbers_ = torch.cat([atomic_numbers, atomic_numbers], dim=0)
        Rij_ = torch.cat([Rij_reactant, Rij_product], dim=0)
        idx_i_ = torch.cat([idx_i, idx_i + offset], dim=0)
        idx_j_ = torch.cat([idx_j, idx_j + offset], dim=0)

        # Get reactant and product embeddings
        q_rp, _ = self._forward(atomic_numbers_, Rij_, idx_i_, idx_j_, type="rp")

        q_r, q_p = torch.split(q_rp, offset, dim=0)

        # Combine reactant and product embeddings
        if self.combine_mode == "diff":
            rp_embedding = q_r - q_p
        elif self.combine_mode == "sum":
            rp_embedding = q_r + q_p
        elif self.combine_mode == "mean":
            rp_embedding = (q_r + q_p) / 2
        elif self.combine_mode == "mlp":
            rp_embedding = self.combine_mlp(torch.cat([q_r, q_p], dim=-1))

        # Get transition state embedding
        q, mu = self._forward(
            atomic_numbers, Rij, idx_i, idx_j, type="ts", rp_embedding=rp_embedding
        )

        # collect results
        inputs["scalar_representation"] = q
        inputs["vector_representation"] = mu

        return inputs

    def _forward(
        self,
        atomic_numbers: torch.Tensor,
        r_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        type: str,
        rp_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute atomic representations/embeddings.

        Args:
            inputs: SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        assert type in [
            "rp",
            "ts",
        ], "type must be either reactant and product (rp) or transition state (ts)"
        n_atoms = atomic_numbers.shape[0]

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
        if self.rp_embedding is not None and type == "rp":
            # own embedding for reactant and product
            q = self.rp_embedding(atomic_numbers)
        else:
            q = self.embedding(atomic_numbers)

        # Combine reactants/product embeddings with ts embedding
        if rp_embedding is not None:
            q = torch.cat([q, rp_embedding], dim=-1)
            q = self.project_scalar_down(q)
        q = q.unsqueeze(1)

        # compute interaction blocks and update atomic embeddings
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
            q = self.dropout(q)
        q = q.squeeze(1)

        return q, mu


class PaiNNRPEmb(PaiNN):
    """PaiNN - with RP embedding"""

    def __init__(
        self,
        combine_mode: str = "diff",
        rp_embedding_size: Optional[int] = None,
        p_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert combine_mode in [
            "diff",
            "sum",
            "mean",
            "mlp",
        ]
        self.combine_mode = combine_mode

        embedding_size_in = (
            2 * self.n_atom_basis
            if rp_embedding_size is None
            else 2 * rp_embedding_size
        )
        embedding_size_out = (
            self.n_atom_basis if rp_embedding_size is None else rp_embedding_size
        )
        if self.combine_mode == "mlp":
            self.combine_mlp = nn.Sequential(
                nn.Linear(embedding_size_in, self.n_atom_basis),
                nn.SiLU(),
                nn.Linear(self.n_atom_basis, embedding_size_out),
            )
        self.project_scalar_down_embedding = nn.Linear(
            embedding_size_out, self.n_atom_basis, bias=False
        )
        self.project_scalar_down = nn.Linear(
            self.n_atom_basis * 2, self.n_atom_basis, bias=False
        )
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        atomic_numbers = inputs[properties.Z]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        Rij = inputs[properties.Rij]

        # Get reactant and product embeddings
        q_r, q_p = inputs["scalar_r"], inputs["scalar_p"]

        # Combine reactant and product embeddings
        if self.combine_mode == "diff":
            rp_embedding = q_r - q_p
        elif self.combine_mode == "sum":
            rp_embedding = q_r + q_p
        elif self.combine_mode == "mean":
            rp_embedding = (q_r + q_p) / 2
        elif self.combine_mode == "mlp":
            rp_embedding = self.combine_mlp(torch.cat([q_r, q_p], dim=-1))

        # Project down to the size of the scalar representation of the TS
        rp_embedding = self.project_scalar_down_embedding(rp_embedding)

        # Get transition state embedding
        q, mu = self._forward(atomic_numbers, Rij, idx_i, idx_j, rp_embedding)

        # collect results
        inputs["scalar_representation"] = q
        inputs["vector_representation"] = mu

        return inputs

    def _forward(
        self,
        atomic_numbers: torch.Tensor,
        r_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rp_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute atomic representations/embeddings.

        Args:
            inputs: SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        n_atoms = atomic_numbers.shape[0]

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

        # Combine reactants/product embeddings with ts embedding
        if rp_embedding is not None:
            q = torch.cat([q, rp_embedding], dim=-1)
            q = self.project_scalar_down(q)
        q = q.unsqueeze(1)

        # compute interaction blocks and update atomic embeddings
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
            q = self.dropout(q)
        q = q.squeeze(1)

        return q, mu
