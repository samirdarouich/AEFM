from typing import Dict

import schnetpack.transform as trn
import torch

from aefm import properties

__all__ = [
    "AllToAllNeighborList",
    "SubgraphMask",
]


class AllToAllNeighborList(trn.NeighborListTransform):
    """
    Calculate a full neighbor list for all atoms in the system.
    Faster than other methods and useful for small systems.
    """

    def __init__(self, object_aware=False):
        # pass dummy large cutoff as all neighbors are connceted
        super().__init__(cutoff=1e8)
        self.object_aware = object_aware

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        Z = inputs[properties.Z]
        R = inputs[properties.R]
        cell = inputs[properties.cell]
        pbc = inputs[properties.pbc]
        conditions_idx_m = inputs.get(properties.conditions_idx_m, None)

        idx_i, idx_j, offset, idx_i_fragment, idx_j_fragment = (
            self._build_neighbor_list(Z, R, cell, pbc, self._cutoff, conditions_idx_m)
        )

        inputs[properties.idx_i] = idx_i.detach()
        inputs[properties.idx_j] = idx_j.detach()
        inputs[properties.offsets] = offset

        # Add fragment indices if object-aware
        if idx_i_fragment is not None and idx_j_fragment is not None:
            inputs[properties.idx_i_fragment] = idx_i_fragment.detach()
            inputs[properties.idx_j_fragment] = idx_j_fragment.detach()
        return inputs

    def _build_neighbor_list(
        self,
        Z,
        positions,
        cell,
        pbc,
        cutoff,
        conditions_idx_m=None,
    ):

        if self.object_aware:
            if conditions_idx_m is None:
                raise ValueError("Object-aware neighbor list requires conditions_mask.")
            # split into fragments
            node_indices = torch.arange(len(conditions_idx_m)).reshape(
                conditions_idx_m.max() + 1, -1
            )

            # Get fully connected within each fragment
            idx_i, idx_j, offset = self._create_fully_connected_list(
                node_indices, positions.dtype
            )

            # Get cross-fragment connections
            idx_i_fragment, idx_j_fragment = self._create_fully_connected_interfragment(
                node_indices, conditions_idx_m
            )
        else:
            n_atoms = Z.shape[0]
            idx_i, idx_j, offset = self._create_fully_connected(
                torch.arange(n_atoms), positions.dtype
            )
            idx_i_fragment, idx_j_fragment = None, None

        return idx_i, idx_j, offset, idx_i_fragment, idx_j_fragment

    def _create_fully_connected_list(self, node_indices_list, dtype):
        idx_i, idx_j, offset = self._create_fully_connected(node_indices_list[0], dtype)

        for i in range(1, len(node_indices_list)):
            idx_i_, idx_j_, offset_ = self._create_fully_connected(
                node_indices_list[i], dtype
            )
            idx_i = torch.cat((idx_i, idx_i_))
            idx_j = torch.cat((idx_j, idx_j_))
            offset = torch.cat((offset, offset_))

        return idx_i, idx_j, offset

    def _create_fully_connected_interfragment(
        self, node_indices_list, conditions_idx_m
    ):
        all_nodes = node_indices_list.flatten()
        num_nodes = all_nodes.size(0)

        # Create index grid for all possible pairs
        idx_grid_i, idx_grid_j = torch.meshgrid(
            torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
        )

        # Mask to exclude self-connections within the same group
        mask = conditions_idx_m[idx_grid_i] != conditions_idx_m[idx_grid_j]

        # Apply mask to get connected indices
        idx_i = all_nodes[idx_grid_i[mask]]
        idx_j = all_nodes[idx_grid_j[mask]]

        # shape should be n_atoms_fragment * n_atoms_other_fragments * n_fragments
        assert (
            idx_i.shape[0]
            == node_indices_list.shape[1]
            * (node_indices_list.shape[1] * (node_indices_list.shape[0] - 1))
            * node_indices_list.shape[0]
        )
        return idx_i, idx_j

    @staticmethod
    def _create_fully_connected(node_indices, dtype):
        n_atoms = len(node_indices)
        idx_i = node_indices.repeat_interleave(n_atoms)
        idx_j = node_indices.repeat(n_atoms)

        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

        offset = torch.zeros(n_atoms * (n_atoms - 1), 3, dtype=dtype)

        return idx_i, idx_j, offset


class SubgraphMask(trn.Transform):
    """
    Construct a subgraph mask for a given set of conditions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if properties.conditions_mask not in inputs:
            subgraph_mask = torch.ones_like(
                inputs["_idx_i"], dtype=inputs[properties.R].dtype
            )
        else:
            conditions_mask = inputs[properties.conditions_mask]
            i, j = inputs["_idx_i"], inputs["_idx_j"]

            non_condition_nodes = torch.where(conditions_mask == 1)[0]
            condition_nodes = torch.where(conditions_mask == 0)[0]

            # Cut all edges between true structure and condition nodes
            ij_mask = torch.isin(i, condition_nodes) & torch.isin(
                j, non_condition_nodes
            )

            ji_mask = torch.isin(i, non_condition_nodes) & torch.isin(
                j, condition_nodes
            )

            subgraph_mask = torch.ones(ij_mask.shape)
            subgraph_mask[ij_mask] = 0.0
            subgraph_mask[ji_mask] = 0.0

            # Cut all edges between different condition structures (if more than 1)
            individual_conditions = condition_nodes.split(non_condition_nodes.shape[0])
            if len(individual_conditions) == 2:
                ij_mask_cond = torch.isin(i, individual_conditions[0]) & torch.isin(
                    j, individual_conditions[1]
                )
                ji_mask_cond = torch.isin(i, individual_conditions[1]) & torch.isin(
                    j, individual_conditions[0]
                )
                subgraph_mask[ij_mask_cond] = 0.0
                subgraph_mask[ji_mask_cond] = 0.0

            # Check if correct number of edges left: (N*N-1) * no_structures
            assert subgraph_mask.sum() == non_condition_nodes.shape[0] * (
                non_condition_nodes.shape[0] - 1
            ) * (1 + len(individual_conditions))

        # Add subgraph mask to inputs
        inputs[properties.subgraph_mask] = subgraph_mask.unsqueeze(1)
        return inputs
