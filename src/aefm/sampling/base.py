import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from ase import Atoms
from schnetpack import properties
from schnetpack import transform as trn
from schnetpack.data.loader import _atoms_collate_fn
from torch import nn

logger = logging.getLogger(__name__)
__all__ = ["Sampler"]


class Sampler:
    """
    Base class for for sampling using a generative model.
    """

    def __init__(
        self,
        prediction_net: Optional[Union[nn.Module, str]],
        invariant: bool = True,
        results_on_cpu: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            prediction_net: Neural network to propagate the samples.
            invariant: invariant: if True, force invariance to E(3) symmetries.
                e.g. For atoms positions this would be to force a zero center.
            results_on_cpu: if True, move the returned results to CPU.
            device: the device to use for denoising.
        """
        self.invariant = invariant
        self.results_on_cpu = results_on_cpu
        self.device = device

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prediction_net = self._load_model(prediction_net)

    def _load_model(self, model: Optional[Union[nn.Module, str]]):
        if isinstance(model, str):
            logger.info(f"Loading model from: '{model}'")
            try:
                model = torch.load(
                    model,
                    map_location=self.device,
                    weights_only=False,
                )
            except AttributeError:
                model = torch.load(
                    model,
                    map_location=self.device,
                )
            if type(model) is dict:
                model = model["hyper_parameters"]["model"]
            model = model.eval()
        elif model is not None:
            logger.info("Using provided model")
            model = model.to(self.device).eval()

        return model

    def _infer_inputs(
        self, system: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Checks the input data and fills missing data with random values,
        for instance, if only atomic numbers Z but no positions are given
        to sample/denoise from p(R|Z).

        Args:
            system: dict containing one input system.
        """
        # check input format
        if not isinstance(system, dict):
            raise ValueError("Inputs must be dicts.")

        # check if all necessary properties are present in the input
        if properties.Z not in system:
            raise NotImplementedError(
                "Atomic numbers must be provided as input."
                " This sampler models the conditional p(R|Z)."
            )

        # get atomic numbers
        numbers = system[properties.Z]

        # get or initialize positions
        if properties.R not in system:
            positions = torch.randn(len(numbers), 3, device=self.device)
        else:
            positions = system[properties.R]

        return numbers, positions

    def prepare_inputs(
        self,
        inputs: List[Union[torch.Tensor, Dict[str, torch.Tensor], Atoms]],
        transforms: Optional[List[trn.Transform]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepares and converts the inputs in SchNetPack format for the sampler.
        Args:
            inputs: the inputs to be converted. Supports:
                        - one element or list of tensors with the atomic numbers Z
                        - one element or list of dicts of tensors including R and Z
                        - one element or list of ase.Atoms
            transforms: Optional transforms to apply to the inputs.
        """
        # set default transforms
        if transforms is None:
            transforms = [
                trn.CastTo64(),
                trn.SubtractCenterOfGeometry(),
            ]

        # check inputs format
        if (
            isinstance(inputs, torch.Tensor)
            or isinstance(inputs, dict)
            or isinstance(inputs, Atoms)
        ):
            inputs = [inputs]
        elif not isinstance(inputs, list):
            raise ValueError(
                "Inputs must be:"
                "one element or list of tensors with the atomic numbers Z "
                "one element or list of Dict of tensors including R and Z "
                "one element or list of ase.Atoms."
            )

        # convert inputs to SchNetPack batch format
        batch = []
        for idx, system in enumerate(inputs):
            if isinstance(system, torch.Tensor):
                system = {properties.Z: system}

            if isinstance(system, dict):
                # sanity checks
                numbers, positions = self._infer_inputs(system)

                # convert to ase.Atoms
                mol = Atoms(numbers=numbers, positions=positions)
            else:
                mol = system
                system = {}

            # convert to dict of tensors in SchNetPack format
            system.update(
                {
                    properties.n_atoms: torch.tensor(
                        [mol.get_global_number_of_atoms()]
                    ),
                    properties.Z: torch.from_numpy(mol.get_atomic_numbers()),
                    properties.R: torch.from_numpy(mol.get_positions()),
                    properties.cell: torch.from_numpy(mol.get_cell().array).view(
                        -1, 3, 3
                    ),
                    properties.pbc: torch.from_numpy(mol.get_pbc()).view(-1, 3),
                    properties.idx: torch.tensor([idx]),
                }
            )

            # apply transforms
            for transform in transforms:
                system = transform(system)

            batch.append(system)

        # collate batch in a dict of tensors in SchNetPack format
        batch = _atoms_collate_fn(batch)

        # Move input batch to device
        batch = {p: batch[p].to(self.device) for p in batch}

        return batch

    def update_model(self, model: nn.Module):
        """
        Updates the velocity model.

        Args:
            model: the new velocity model.
        """
        if next(model.parameters()).device != self.device:
            model = model.to(self.device)
        self._model_state = model.training
        self.prediction_net = model.eval()

    def restore_model(self):
        """
        Restores the velocity model to its original state.
        """
        self.prediction_net.training = self._model_state

    def sample(
        self,
        inputs: Dict[str, torch.Tensor],
        t: torch.Tensor,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Peforms sampling using the learned prediction net. Returns the sampled data at
        times specified by t.

        Args:
            inputs: dict with input data in the SchNetPack form, inluding the starting
                x_t.
            t: the time steps to sample.
            kwargs: additional keyword arguments.
        """
        raise NotImplementedError

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """
        Defines the default call method.
        Currently equivalent to calling ``self.sample``.
        """
        return self.sample(*args, **kwargs)
