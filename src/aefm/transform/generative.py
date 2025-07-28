import copy
import logging
from typing import Dict, Union

import schnetpack
import schnetpack.transform as trn
import torch
import torch.nn as nn
from schnetpack.utils import load_model
from torch_scatter import scatter_mean
from aefm import properties
from aefm.processes import DiffusionProcess, FlowProcess
from aefm.processes.functional import (
    sample_noise_like,
)
from aefm.utils.analysis import _ase_align

log = logging.getLogger(__name__)

__all__ = [
    "ConditionalFlow",
    "Diffuse",
    "OneShot",
    "OneShotPrior",
]


class ConditionalFlow(trn.Transform):
    """
    Wrapper class for conditional flow process of molecular properties.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        flow_property: str,
        flow_process: FlowProcess,
        output_key: str,
        time_key: str = "t",
    ):
        """
        Args:
            flow_property: the property to aefm.
            flow_process: the flow matching process to use.
            output_key: key to save the flow velocity.
            time_key: key to save the flow time step.
        """
        super().__init__()
        self.flow_property = flow_property
        self.flow_process = flow_process
        self.output_key = output_key
        self.time_key = time_key

        # Sanity check
        if not self.flow_process.invariant and self.flow_property == properties.R:
            logging.error(
                "Flow of atom positions R without invariant constraint"
                "(invariant=False) might lead to unexpected results."
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Define the flow transformation for a unique sample.

        Args:
            inputs: dictionary of input tensors as in SchNetPack.
        """
        # Get flow property for sample x_0 and x_1 (no conditions included)
        x_0 = inputs[properties.x_0]
        x_1 = inputs[properties.x_1]
        device = x_0.device

        # Save the original flow property (possibly with conditioning values)
        original_flow_prop = inputs[self.flow_property].clone()
        outputs = {}

        # Get idx_m for sample
        idx_m = inputs.get(properties.idx_m, torch.zeros_like(inputs[properties.Z]))

        # Sample training time step for the input molecule (the same for the conditions)
        t = torch.rand(
            size=(idx_m.max().item() + 1,),
            device=device,
        )[idx_m]

        # Add time step and broadcast to all atoms (also for conditioning atoms)
        outputs[self.time_key] = t

        # If x_0 has the same shapen as idx_m, then joint distribution is learned
        flag_combined = idx_m.shape[0] == x_0.shape[0]

        # If not combined only flow non conditioning atoms
        if not flag_combined:
            condition_mask = inputs[properties.conditions_mask]
            idx_m = idx_m[condition_mask == 1]
            t = t[condition_mask == 1]

        # In case joint distribution is modeled, conditions_idx_m needs to be used
        if properties.conditions_idx_m in inputs and flag_combined:
            idx_m = inputs[properties.conditions_idx_m]

        # Flow the property.
        tmp = self.flow_process.flow(
            x_0,
            x_1,
            idx_m=idx_m,
            t=t,
            return_dict=True,
            sample_key=self.flow_property,
            output_key=self.output_key,
        )
        outputs.update(tmp)

        # Add conditiong values on diffused property and set time = 0.0 for conditioning
        # atoms
        if not flag_combined:
            outputs[self.time_key][condition_mask == 0] = 0.0

            # Overwrite non conditiong atoms with flowed property
            original_flow_prop[condition_mask == 1] = outputs[self.flow_property]
            outputs[self.flow_property] = original_flow_prop

            # Add 0s for output (is anyway ignored with constraint mask)
            velocity = torch.zeros_like(original_flow_prop)
            velocity[condition_mask == 1] = outputs[self.output_key]
            outputs[self.output_key] = velocity

        # update the returned inputs.
        inputs.update(outputs)
        rmsd = scatter_mean(((x_1-outputs[self.flow_property])**2).sum(-1), idx_m, dim=0).sqrt()
        inputs[properties.rmsd+"_intermediate"] = rmsd

        return inputs


class Diffuse(trn.Transform):
    """
    Wrapper class for diffusion process of molecular properties.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        diffuse_property: str,
        diffusion_process: DiffusionProcess,
        output_key: str,
        time_key: str = "t",
    ):
        """
        Args:
            diffuse_property: molecular property to diffuse.
            diffusion_process: the forward diffusion process to use.
            output_key: key to store the noise.
            time_key: key to save the normalized diffusion time step.
        """
        super().__init__()
        self.diffuse_property = diffuse_property
        self.diffusion_process = diffusion_process
        self.output_key = output_key
        self.time_key = time_key

        # Sanity check
        if (
            not self.diffusion_process.invariant
            and self.diffuse_property == properties.R
        ):
            logging.error(
                "Diffusing atom positions R without invariant constraint"
                "(invariant=False) might lead to unexpected results."
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Define the forward diffusion transformation.

        Args:
            inputs: dictionary of input tensors as in SchNetPack.
        """
        # Get diffusion property for sample
        x_0 = inputs[properties.x_0]
        device = x_0.device

        # Save the original diffusion property (possibly with conditioning values)
        original_diffuse_prop = inputs[self.diffuse_property].clone()
        outputs = {}

        # Get idx_m for sample
        idx_m = inputs.get(properties.idx_m, torch.zeros_like(inputs[properties.Z]))

        # Sample training time step for the input molecule (the same for the conditions)
        # and broadcast it for each atom.
        t = torch.randint(
            0,
            self.diffusion_process.get_T(),
            size=(idx_m.max().item() + 1,),
            dtype=torch.long,
            device=device,
        )[idx_m]

        # Add time step
        outputs[self.time_key] = t

        # normalize the time step to [0,1].
        outputs[self.time_key] = self.diffusion_process.normalize_time(
            outputs[self.time_key]
        )

        # If x_0 has the same shapen as idx_m, then joint distribution is learned
        flag_combined = idx_m.shape[0] == x_0.shape[0]

        # If not combined only flow non conditioning atoms
        if not flag_combined:
            condition_mask = inputs[properties.conditions_mask]
            idx_m = idx_m[condition_mask == 1]
            t = t[condition_mask == 1]

        # In case joint distribution is modeled, conditions_idx_m needs to be used
        if properties.conditions_idx_m in inputs and flag_combined:
            idx_m = inputs[properties.conditions_idx_m]

        # diffuse the property.
        tmp = self.diffusion_process.diffuse(
            x_0,
            idx_m=idx_m,
            t=t,
            return_dict=True,
            sample_key=self.diffuse_property,
            output_key=self.output_key,
        )
        outputs.update(tmp)

        # Add conditiong values on diffused property and set time = 0.0 for conditioning
        # atoms
        if not flag_combined:
            outputs[self.time_key][condition_mask == 0] = 0.0

            # Overwrite non conditiong atoms with flowed property
            original_diffuse_prop[condition_mask == 1] = outputs[self.diffuse_property]
            outputs[self.diffuse_property] = original_diffuse_prop

            # Add 0s for output (is anyway ignored with constraint mask)
            noise = torch.zeros_like(original_diffuse_prop)
            noise[condition_mask == 1] = outputs[self.output_key]
            outputs[self.output_key] = noise

        # update the returned inputs.
        inputs.update(outputs)

        return inputs


class OneShot(trn.Transform):
    """
    Wrapper class for conditional flow process of molecular properties.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        property: str,
        output_key: str,
        sigma: float = 0.0,
    ):
        """
        Args:
            property: the property to aefm.
            flow_process: the flow matching process to use.
            output_key: key to save the flow velocity.
            time_key: key to save the flow time step.
        """
        super().__init__()
        self.property = property
        self.output_key = output_key
        self.sigma = sigma

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Define the flow transformation for a unique sample.

        Args:
            inputs: dictionary of input tensors as in SchNetPack.
        """
        # Get flow property for sample x_0 and x_1 (no conditions included)
        x_0 = inputs[properties.x_0]
        x_1 = inputs[properties.x_1]

        # Save the original flow property (possibly with conditioning values)
        original_flow_prop = inputs[self.property]
        outputs = {}

        # In case a joint distribution should be learned, provide all structures at once
        idx_m = None
        flag_combined = False
        if properties.conditions_idx_m in inputs:
            if inputs[properties.conditions_idx_m].shape[0] == x_0.shape[0]:
                flag_combined = True
                idx_m = inputs[properties.conditions_idx_m]

        # Predict the target structure x1
        outputs[self.output_key] = x_1.clone().detach()

        # If wanted add noise to input structure
        outputs[self.property] = original_flow_prop.clone().detach()

        if self.sigma > 0.0:
            eps = sample_noise_like(x_0, invariant=True, idx_m=idx_m)
            outputs[self.property] = outputs[self.property] + self.sigma * eps

        if properties.conditions_mask in inputs and not flag_combined:
            condition_mask = inputs[properties.conditions_mask] == 1

            # Add 0s for output (velocity)
            target = torch.zeros_like(original_flow_prop)
            target[condition_mask] = outputs[self.output_key]
            outputs[self.output_key] = target

        # update the returned inputs.
        inputs.update(outputs)

        return inputs


class OneShotPrior(trn.Transform):

    def __init__(
        self,
        model: Union[str, schnetpack.model.AtomisticModel],
        output_key: str,
        model_output_key: str,
        sigma: float = 0.0,
    ):
        super().__init__()
        self.model = self._load_model(model, device="cpu")
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.sigma = sigma

    def _load_model(
        self, model_file: Union[str, schnetpack.model.AtomisticModel], device: str
    ) -> schnetpack.model.AtomisticModel:
        """
        Load an individual model, activate stress computation

        Args:
            model_file (str): path to model.

        Returns:
           AtomisticTask: loaded schnetpack model
        """

        if isinstance(model_file, str):
            log.info("Loading model from {:s}".format(model_file))

            model = load_model(model_file, device=device)
        elif isinstance(model_file, schnetpack.model.AtomisticModel):
            model = model_file.to(device=device)
        else:
            raise ValueError("Invalid model type.")

        model = model.eval()

        # Check for postprocessors that cast to double precision and remove them
        model.postprocessors = nn.ModuleList(
            [
                postprocessor
                for postprocessor in model.postprocessors
                if not isinstance(postprocessor, schnetpack.transform.CastTo64)
            ]
        )

        return model

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        model_inputs = copy.deepcopy(inputs)

        if properties.idx_m not in model_inputs:
            model_inputs[properties.idx_m] = torch.repeat_interleave(
                torch.arange(len(model_inputs[properties.n_atoms])),
                repeats=model_inputs[properties.n_atoms],
                dim=0,
            )

        with torch.no_grad():
            model_outputs = self.model(model_inputs)

        outputs = {self.output_key: model_outputs[self.model_output_key]}

        # Align output
        x_0 = outputs[self.output_key]
        x_1 = inputs[properties.x_1]

        # Add noise to prior output
        idx_m = None
        if properties.conditions_idx_m in inputs:
            if inputs[properties.conditions_idx_m].shape[0] == x_0.shape[0]:
                idx_m = inputs[properties.conditions_idx_m]

        if self.sigma > 0.0:
            eps = sample_noise_like(x_0, invariant=True, idx_m=idx_m)
            x_0 = x_0 + self.sigma * eps

        # Add additonal noise
        # Overwrite conditioning atoms with the original values
        if properties.conditions_mask in inputs:
            if inputs[properties.conditions_mask].shape[0] == x_0.shape[0]:

                conditions_mask = inputs[properties.conditions_mask]

                # Align with x_1
                outputs[self.output_key][conditions_mask == 1] = _ase_align(
                    x_0[conditions_mask == 1], x_1[conditions_mask == 1]
                )

                # Overwrite prediction with conditioning values
                outputs[self.output_key][conditions_mask == 0] = inputs[
                    self.output_key
                ][conditions_mask == 0]
        else:
            outputs[self.output_key] = _ase_align(x_0, x_1)

        # Save the prediction and original prior as well
        outputs[properties.x_0] = outputs[self.output_key].clone().detach()
        inputs[properties.x_0 + "_orig"] = inputs[properties.x_0].clone().detach()

        # Overwrite with prior
        inputs.update(outputs)

        return inputs
