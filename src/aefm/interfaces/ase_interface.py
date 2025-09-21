from copy import deepcopy
from schnetpack.interfaces.ase_interface import AtomsConverterError
from typing import Optional, List, Union, Dict
from ase import Atoms
from schnetpack.data.loader import _atoms_collate_fn
from aefm import properties
from omegaconf import ListConfig
from schnetpack.transform import CastTo32, CastTo64, Transform
import torch

__all__ = ["ReactionConverter"]

class ReactionConverter:
    """
    Convert ASE atoms to SchNetPack input batch format for model prediction. This 
    works with reaction conditioning via atoms.arrays

    """

    def __init__(
        self,
        neighbor_list: Union[Transform, None],
        transforms: Optional[Union[Transform, List[Transform]]] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        additional_inputs: Optional[Dict[str, torch.Tensor]] = None,
        conditioning_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            neighbor_list (Transform, None): neighbor list transform. Can be set to None incase
                that the neighbor list is contained in transforms.
            transforms: transforms for manipulating the neighbor lists. This can be either a single transform or a list
                of transforms that will be executed after the neighbor list is calculated. Such transforms may be
                useful, e.g., for filtering out certain neighbors. In case transforms are required before the neighbor
                list is calculated, neighbor_list argument can be set to None and a list of transforms including the
                neighbor list can be passed as transform argument. The transforms will be executed in the order of
                their appearance in the list.
            device (str, torch.device): device on which the model operates (default: cpu).
            dtype (torch.dtype): required data type for the model input (default: torch.float32).
            additional_inputs (dict): additional inputs required for some transforms.
                When setting up the AtomsConverter, those additional inputs will be
                stored to the input batch.
            conditioning_keys: List of keys in atoms.arrays that should be used as
                conditions. If None, no conditions are added.
        """

        self.neighbor_list = deepcopy(neighbor_list)
        self.device = device
        self.dtype = dtype
        self.additional_inputs = additional_inputs or {}
        self.conditioning_keys = conditioning_keys or []
        
        # convert transforms and neighbor_list to list
        transforms = transforms or []
        if isinstance(transforms,ListConfig):
            transforms = list(transforms)
        if not isinstance(transforms, list):
            transforms = [transforms]
        neighbor_list = [] if neighbor_list is None else [neighbor_list]

        # get transforms and initialize neighbor list
        self.transforms: List[Transform] = neighbor_list + transforms

        # Set numerical precision
        if dtype == torch.float32:
            self.transforms.append(CastTo32())
        elif dtype == torch.float64:
            self.transforms.append(CastTo64())
        else:
            raise AtomsConverterError(f"Unrecognized precision {dtype}")

    def _add_conditions(
        self,
        inputs: Dict[str, torch.Tensor],
        conditions: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Add conditions to input."""

        # Add conditions mask
        inputs[properties.conditions_mask] = torch.ones_like(inputs[properties.Z])

        # Start conditions count with 0 for real structure
        inputs[properties.conditions_idx_m] = torch.zeros_like(inputs[properties.Z])

        # Loop throuh conditions
        for i, condition in enumerate(conditions):
            # Create mask with 0s for conditioning atoms
            inputs[properties.conditions_mask] = torch.cat(
                [
                    inputs[properties.conditions_mask],
                    torch.zeros_like(condition[properties.Z]),
                ],
                dim=0,
            )

            # Add condition count
            inputs[properties.conditions_idx_m] = torch.cat(
                [
                    inputs[properties.conditions_idx_m],
                    torch.ones_like(condition[properties.Z]) * (i + 1),
                ],
                dim=0,
            )

            # Concat 3D conditions to input
            n_atoms_input = inputs[properties.n_atoms]
            n_atoms_condition = condition[properties.n_atoms]
            inputs[properties.n_atoms] = n_atoms_input + n_atoms_condition
            inputs[properties.Z] = torch.cat(
                [inputs[properties.Z], condition[properties.Z]], dim=0
            )
            inputs[properties.R] = torch.cat(
                [inputs[properties.R], condition[properties.R]], dim=0
            )

        return inputs
    
    def __call__(self, atoms: List[Atoms] or Atoms):
        """

        Args:
            atoms (list or ase.Atoms): list of ASE atoms objects or single ASE atoms object.

        Returns:
            dict[str, torch.Tensor]: input batch for model.
        """

        # check input type and prepare for conversion
        if isinstance(atoms,list):
            pass
        elif isinstance(atoms, Atoms):
            atoms = [atoms]
        else:
            raise TypeError(
                "atoms is type {}, but should be either list or ase.Atoms object".format(
                    type(atoms)
                )
            )

        inputs_batch = []
        for at_idx, at in enumerate(atoms):

            inputs = {
                properties.n_atoms: torch.tensor([at.get_global_number_of_atoms()]),
                properties.Z: torch.from_numpy(at.get_atomic_numbers()),
                properties.R: torch.from_numpy(at.get_positions()),
                properties.cell: torch.from_numpy(at.get_cell().array).view(-1, 3, 3),
                properties.pbc: torch.from_numpy(at.get_pbc()).view(-1, 3),
            }

            # specify sample index
            inputs.update({properties.idx: torch.tensor([at_idx])})

            # add additional inputs (specified in AtomsConverter __init__)
            inputs.update(self.additional_inputs)

            # add conditions if available
            conditions = []
            for key in self.conditioning_keys:
                condition_pos = at.arrays.get(key, None)
                if condition_pos is None:
                    raise KeyError(f"Conditioning key '{key}' not found in atoms.arrays")
                assert condition_pos.shape == inputs[properties.R].shape, "Conditioning positions shape does not match atoms positions shape"
                conditions.append(
                    {
                        properties.n_atoms: torch.tensor([at.get_global_number_of_atoms()]),
                        properties.Z: torch.from_numpy(at.get_atomic_numbers()),
                        properties.R: torch.from_numpy(condition_pos),
                        properties.cell: torch.from_numpy(at.get_cell().array).view(-1, 3, 3),
                        properties.pbc: torch.from_numpy(at.get_pbc()).view(-1, 3),
                    }
                )
            if len(conditions) > 0:
                inputs = self._add_conditions(inputs, conditions)

            for transform in self.transforms:
                inputs = transform(inputs)
            inputs_batch.append(inputs)

        inputs = _atoms_collate_fn(inputs_batch)

        # Move input batch to device
        inputs = {p: inputs[p].to(self.device) for p in inputs}

        return inputs