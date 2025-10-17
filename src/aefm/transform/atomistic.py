import logging
from typing import Dict

import schnetpack.transform as trn
import torch
from schnetpack.transform import (
    AddOffsets as SnnAddOffsets,
)
from schnetpack.transform import (
    RemoveOffsets as SnnRemoveOffsets,
)
from torch_scatter import scatter_add

from aefm import properties
from aefm.utils import batch_center_systems

log = logging.getLogger(__name__)

__all__ = [
    "BatchSubtractCenterOfGeometry",
    "SubtractCenterOfGeometry",
    "ComputeDistances",
    "RemoveOffsets",
    "AddOffsets",
    "Convert2PyG",
]


class BatchSubtractCenterOfGeometry(trn.Transform):
    """
    Subsctract center of geometry from input systems batchwise.
    """

    is_preprocessor: bool = False
    is_postprocessor: bool = True
    force_apply: bool = True

    def __init__(
        self,
        name: str = "v_t_pred",
        dim: int = 3,
    ):
        """
        Args:
            name: name of the property to be centered.
            dim: number of dimensions of the property to be centered.
        """
        super().__init__()
        self.name = name
        self.dim = dim

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        forward pass of the transform.

        Args:
            inputs: dictionary of input tensors.
        """
        # check shapes
        if inputs[self.name].shape[1] < self.dim:
            raise ValueError(
                f"Property {self.name} has less than {self.dim} dimensions. "
                f"Cannot subtract center of mass."
            )

        # Check if conditions are present (important for 3D constraints)
        idx_m = inputs[properties.idx_m]
        if properties.conditions_idx_m in inputs:
            idx_m = inputs[properties.conditions_idx_m]

        # center batchwise
        if inputs[self.name].shape[-1] == self.dim:
            inputs[self.name] = batch_center_systems(
                inputs[self.name],
                idx_m,
            )
        # use the first dimensions if the property has more than 'dim' dimensions.
        else:
            x = inputs[self.name][:, : self.dim]
            h = inputs[self.name][:, self.dim :]

            # Check if conditions are present (important for 3D constraints)
            x_cent = batch_center_systems(
                x,
                idx_m,
            )
            inputs[self.name] = torch.cat((x_cent, h), dim=-1).to(
                device=inputs[self.name].device
            )

        return inputs


class SubtractCenterOfGeometry(trn.Transform):
    """
    Subtract center of geometry from positions. Treating the conditions seperately
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if properties.conditions_mask in inputs:
            # Remove center of geometry from each structure (including conditioning
            # structures)
            inputs[properties.position] = batch_center_systems(
                inputs[properties.position], inputs[properties.conditions_idx_m]
            )
        else:
            inputs[properties.position] -= inputs[properties.position].mean(0)
        return inputs


class ComputeDistances(trn.Transform):
    is_preprocessor: bool = True
    is_postprocessor: bool = True
    force_apply: bool = True

    def __init__(
        self,
        name: str = "_positions",
    ):
        """
        Args:
            name: name of the property for which the distances should be computed.
        """
        super().__init__()
        self.name = name
        self.model_outputs = [self.name + "_distances"]

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        if properties.idx_i and properties.idx_j in inputs:
            # Extract edge indices for distance computation
            idx_i = inputs[properties.idx_i]
            idx_j = inputs[properties.idx_j]
        else:
            # Construct fully connected distance matrix for each molecule
            num_nodes_each_mol = inputs.get(
                properties.conditions_n_atoms, inputs[properties.n_atoms]
            )

            seg_m = torch.cumsum(num_nodes_each_mol, dim=0)
            seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)

            idx_i, idx_j = torch.cat(
                [
                    torch.combinations(torch.arange(n, device=seg_m.device)) + offset
                    for n, offset in zip(num_nodes_each_mol, seg_m)
                ]
            ).T

        # Compute distances
        r_ij = inputs[self.name][idx_i] - inputs[self.name][idx_j]
        inputs[self.model_outputs[0]] = torch.norm(r_ij, dim=-1)

        return inputs


class RemoveOffsets(SnnRemoveOffsets):
    """
    Remove offsets from property based on the mean of the training data and/or the
    single atom reference calculations.

    The `mean` and/or `atomref` are automatically obtained from the AtomsDataModule,
    when it is used. Otherwise, they have to be provided in the init manually.
    """

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.remove_mean:
            n_atoms = inputs[properties.n_atoms]
            if properties.conditions_n_atoms in inputs:
                n_atoms = inputs[properties.conditions_n_atoms]
            mean = self.mean * n_atoms if self.is_extensive else self.mean
            inputs[self._property] -= mean
        if self.remove_atomrefs:
            if properties.conditions_idx_m in inputs:
                v0 = scatter_add(
                    self.atomref[inputs[properties.Z]],
                    inputs[properties.conditions_idx_m],
                )
            else:
                v0 = torch.sum(self.atomref[inputs[properties.Z]])
            inputs[self._property] -= v0

        return inputs


class AddOffsets(SnnAddOffsets):
    """
    Add offsets to property based on the mean of the training data and/or the single
    atom reference calculations.

    The `mean` and/or `atomref` are automatically obtained from the AtomsDataModule,
    when it is used. Otherwise, they have to be provided in the init manually.

    Hint:
        Place this postprocessor after casting to float64 for higher numerical
        precision.
    """

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.add_mean:
            n_atoms = inputs[properties.n_atoms]
            if properties.conditions_n_atoms in inputs:
                n_atoms = inputs[properties.conditions_n_atoms]
            mean = self.mean * n_atoms if self.is_extensive else self.mean
            inputs[self._property] += mean

        if self.add_atomrefs:
            if properties.conditions_idx_m in inputs:
                y0 = scatter_add(
                    self.atomref[inputs[properties.Z]],
                    inputs[properties.conditions_idx_m],
                )
                n_atoms = inputs[properties.conditions_n_atoms]
            else:
                y0 = scatter_add(
                    self.atomref[inputs[properties.Z]], inputs[properties.idx_m]
                )
                n_atoms = inputs[properties.n_atoms]

            if not self.is_extensive:
                y0 /= n_atoms

            inputs[self._property] += y0

        return inputs

class Convert2PyG(trn.Transform):
    """
    Convert data to PyG format.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        from torch_geometric.data import Data

        pyg_data = Data()

        for key, value in inputs.items():
            pyg_data[key] = value

        pyg_data = Data(
            pos=inputs[properties.R],
            atomic_numbers=inputs[properties.Z].long(),
            batch=inputs[properties.idx_m].long(),
            natoms=inputs[properties.n_atoms].long(),
            edge_index=torch.stack(
                [inputs[properties.idx_j], inputs[properties.idx_i]], dim=0
            ).long(),
            original_inputs=inputs,
            # switch j and i definition
        )

        return pyg_data