import warnings
from typing import Dict, List

import numpy as np
import ot as pot
import schnetpack.transform as trn
import torch
from ase.mep.neb import idpp_interpolate

from aefm import properties
from aefm.processes.functional import (
    sample_noise_like,
)
from aefm.utils.analysis import _ase_align, inputs_to_atoms

__all__ = [
    "ReactionTransform",
    "GenerativeStrategy",
    "AdaptivePrior",
    "FlowTSinitTS",
    "DiffusionTS",
]


def _analyze_inputs(inputs: List[Dict[str, torch.Tensor]]):
    """
    Analyzes the input list of dictionaries to identify and categorize reactants,
    products, transition states, and intermediates based on their image types.

    Args:
        inputs:: A list of dictionaries where each dictionary contains properties of a
            chemical species, including its image type.
    Returns:
        A tuple containing the reactant, product, transition state, and a list of
        final intermediates between R and TS, between P and TS, and all not final
        optimized intermediates.

    """
    reactant = None
    product = None
    transition_state = None
    ts_flag = False
    r_intermediates = []
    p_intermediates = []
    all_intermediates = []
    transition_state_priors = None
    for inp in inputs:
        image_type = inp[properties.image_type].item()

        if image_type == properties.IMAGE_TYPES["reactant"]:
            reactant = inp
        elif image_type == properties.IMAGE_TYPES["product"]:
            product = inp
        elif image_type == properties.IMAGE_TYPES["transition_state"]:
            transition_state = inp
            ts_flag = True
        elif image_type == properties.IMAGE_TYPES["intermediate_final"]:
            if ts_flag:
                p_intermediates.append(inp)
            else:
                r_intermediates.append(inp)
        elif image_type == properties.IMAGE_TYPES["intermediate"]:
            all_intermediates.append(inp)
        elif image_type == properties.IMAGE_TYPES["transition_state_prior"]:
            transition_state_priors = inp
        else:
            raise ValueError(f"Unknown image type: {image_type}")

    # Assert that reactant and transition state are not None, or product and
    # transition state are not None
    if reactant is None and transition_state is None and product is None:
        raise ValueError(
            "Either reactant, transition state or product must be provided."
        )

    return (
        reactant,
        product,
        transition_state,
        r_intermediates,
        p_intermediates,
        all_intermediates,
        transition_state_priors,
    )


class ReactionTransform(trn.Transform):
    """Reaction transform to prepare input from a reaction input."""

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Override with specific reaction transform implementation"""
        raise NotImplementedError


class GenerativeStrategy(ReactionTransform):
    """Reaction transform for generative models. This will prepare the reaction input to
    be used in a flow model.
    """

    def __init__(
        self,
        target_property: str = properties.R,
        sigma_x_0: float = 0.0,
        sigma_x_1: float = 0.0,
        conditioned: bool = False,
        align: bool = False,
        add_conditions_in_target: bool = False,
    ):
        """
        Args:
            target_property: The target property to be predicted by the flow model.
            sigma_x_0: Standard deviation of the noise added to x_0.
            sigma_x_1: Standard deviation of the noise added to x_1.
            conditioned: Whether to condition the flow model on reactant and product.
            align: Whether to align x_0 to x_1.
            add_conditions_in_target: Whether to add conditions in the target property.
                    This is used for inpainting, where the joint distribution instead
                    of the conditioned distribution is learned.
        """
        super().__init__()
        self.target_property = target_property
        self.sigma_x_0 = sigma_x_0
        self.sigma_x_1 = sigma_x_1
        self.conditioned = conditioned
        self.align = align
        self.add_conditions_in_target = add_conditions_in_target

    def _add_conditions(
        self,
        inputs: Dict[str, torch.Tensor],
        conditions: List[Dict[str, torch.Tensor]],
        additional_properties: List[str] = [],
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

            # Add conditions to x_0 and x_1 and target if needed
            if self.add_conditions_in_target:

                # Add noise if wanted to source and target
                x_0_condition = self._add_noise(
                    condition[self.target_property], self.sigma_x_0
                )
                x_1_condition = self._add_noise(
                    condition[self.target_property], self.sigma_x_1
                )

                inputs[properties.x_0] = torch.cat(
                    [inputs[properties.x_0], x_0_condition], dim=0
                )

                if properties.x_1 in inputs:
                    inputs[properties.x_1] = torch.cat(
                        [inputs[properties.x_1], x_1_condition], dim=0
                    )

                inputs[f"target_{self.target_property}"] = torch.cat(
                    [
                        inputs[f"target_{self.target_property}"],
                        condition[self.target_property],
                    ],
                    dim=0,
                )

                for additional_property in additional_properties:
                    inputs[additional_property] = torch.cat(
                        [inputs[additional_property], condition[additional_property]],
                        dim=0,
                    )

            else:
                # Add dummy 0's to target to have same shape
                inputs[f"target_{self.target_property}"] = torch.cat(
                    [
                        inputs[f"target_{self.target_property}"],
                        torch.zeros_like(inputs[f"target_{self.target_property}"])[
                            :n_atoms_condition
                        ],
                    ],
                    dim=0,
                )

        # If conditions are in target, change the value of the target property to x_0
        if self.add_conditions_in_target:
            inputs[self.target_property] = inputs[properties.x_0]

        return inputs

    def _add_noise(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Add noise to input if sigma > 0.0.

        Args:
            x: Input tensor.
            sigma: Standard deviation of the noise.

        Returns:
            x with added noise.
        """
        if sigma > 0.0:
            eps = sample_noise_like(x, invariant=True, idx_m=None)
            x = x + sigma * eps
        return x

    def _get_map(self, x_0, x_1, method: str = "emd"):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x_0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x_1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        method : str
            represents the method to sample from the OT plan

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        # assign uniform importance to all samples
        a, b = pot.unif(x_0.shape[0]), pot.unif(x_1.shape[0])
        if x_0.dim() > 2:
            x_0 = x_0.reshape(x_0.shape[0], -1)
        if x_1.dim() > 2:
            x_1 = x_1.reshape(x_1.shape[0], -1)
        M = torch.cdist(x_0, x_1) ** 2
        if method == "emd":
            p = pot.emd(a, b, M.detach().cpu().numpy())
        elif method == "sinkhorn":
            p = pot.sinkhorn(a, b, M.detach().cpu().numpy(), 0.05)
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x_0, x_1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def _sample_map(self, pi, batch_size, replace=True):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        batch_size : int
            represents the number of samples
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])


class AdaptivePrior(GenerativeStrategy):
    """ 
    Adaptive prior for AEFM. This will prepare the source to be a noised version
    of the ground truth transition state and models the flow to the transition state.
    """

    def __init__(
        self,
        sigma: float,
        target_property: str = properties.R,
        sigma_x_1: float = 0.0,
        align: bool = False,
        conditioned: bool = False,
        add_conditions_in_target: bool = False,
    ):
        """
        Args:
            sigma: Standard deviation of the noise added to x_1 to mimic low-fidelity
                    error.
            target_property: The target property to be predicted by the flow model.
                    Typically the cartesian coordinates of the atoms.
            sigma_x_1: Standard deviation of the noise added to x_1. Can improve
                    stability.
            align: Whether to align x_0 to x_1. Might be helpful when sigma is large.
            conditioned: Whether to condition the flow model on reactant and product.
            add_conditions_in_target: Whether to add conditions in the target property.
                    This is used for inpainting, where the joint distribution instead
                    of the conditioned distribution is learned.
        """
        super().__init__(
            target_property=target_property,
            sigma_x_0=0.0,
            sigma_x_1=sigma_x_1,
            align=align,
            conditioned=conditioned,
            add_conditions_in_target=add_conditions_in_target,
        )
        self.sigma = sigma
        
    def forward(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        # Group inputs by image type
        reactant, product, transition_state, *_ = _analyze_inputs(inputs)

        assert transition_state is not None, "Transition state must be provided."

        # Get source as transition state + random noise
        x_1 = transition_state[self.target_property]
        x_0 = self._add_noise(x_1, self.sigma)
        
        # If wanted add noise to x_1 (can improve stability)
        x_1 = self._add_noise(x_1, self.sigma_x_1)

        # align x_0 to x_1 (if wanted, can be helpful when sigma is large)
        if self.align:
            x_0 = _ase_align(x_0, x_1)

        output = {
            properties.x_0: x_0,
            properties.x_1: x_1,
            f"target_{self.target_property}": transition_state[self.target_property],
            **transition_state,
        }
        output[self.target_property] = output[properties.x_0]

        # starting point is intermedate
        output[properties.image_type] = torch.tensor(
            [properties.IMAGE_TYPES["intermediate_noise"]]
        )
        
        if self.conditioned:
            # If conditions are present add to input
            output = self._add_conditions(output, [reactant, product])
        
        rmsd = ((x_1-x_0)**2).sum(-1).mean().sqrt()
        output[properties.rmsd+"_initial"] = rmsd.unsqueeze(0)
        outputs = [output]

        return outputs
    
class FlowTSinitTS(GenerativeStrategy):
    """
    Flow matching reaction transform. This will prepare the source to be an inital guess
    of the transition state (based on reactant and product) and model the flow to the
    transition state.
    """

    def __init__(
        self,
        init_type: str = "average",
        add_rp_modeling: bool = False,
        embed_rp: bool = False,
        no_ot_samples: int = 0,
        ot_method: str = "emd",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.init_type = init_type
        self.add_rp_modeling = add_rp_modeling
        self.embed_rp = embed_rp
        self.no_ot_samples = no_ot_samples
        self.ot_method = ot_method
        assert ot_method in [
            "emd",
            "sinkhorn",
        ], "OT method must be either 'emd' or 'sinkhorn'"
        if no_ot_samples > 0:
            assert self.sigma_x_1 > 0, "OT noise only works with noise on target"

    def forward(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        # Group inputs by image type
        reactant, product, transition_state, *_ = _analyze_inputs(inputs)

        # Define x_0 and x_1 for flow
        if self.init_type == "average":
            x_0 = (reactant[self.target_property] + product[self.target_property]) / 2
        elif self.init_type == "idpp":
            r = inputs_to_atoms(reactant)
            p = inputs_to_atoms(product)
            guess = r.copy()
            idpp_interpolate([r, guess, p], traj=None, log=None)
            x_0 = torch.from_numpy(guess.get_positions())
        x_1 = transition_state[self.target_property]

        if self.no_ot_samples > 0:
            # Search OT plan between x_0 and x_1
            x_0 = x_0.repeat(self.no_ot_samples, 1, 1)
            x_1 = x_1.repeat(self.no_ot_samples, 1, 1)
            eps_x_0 = sample_noise_like(x_0, invariant=True, idx_m=None)
            eps_x_1 = sample_noise_like(x_1, invariant=True, idx_m=None)

            x_0 = x_0 + self.sigma_x_0 * eps_x_0
            x_1 = x_1 + self.sigma_x_1 * eps_x_1

            # Get OT plan
            pi = self._get_map(x_0, x_1, method=self.ot_method)
            i, j = self._sample_map(pi, x_0.shape[0], replace=False)

            outputs = []

            for ii, jj in zip(i, j):

                source = x_0[ii]
                target = x_1[jj]

                if self.align:
                    source = _ase_align(source, target)

                output = {
                    properties.x_0: source,
                    properties.x_1: target,
                    f"target_{self.target_property}": transition_state[
                        self.target_property
                    ],
                    **transition_state,
                }
                output[self.target_property] = output[properties.x_0]

                # starting point is intermedate
                output[properties.image_type] = torch.tensor(
                    [properties.IMAGE_TYPES["intermediate"]]
                )

                if self.conditioned:
                    # If conditions are present add to input
                    output = self._add_conditions(output, [reactant, product])

                outputs.append(output)

        else:

            # If wanted add noise to x_0 and x_1
            x_0 = self._add_noise(x_0, self.sigma_x_0)
            x_1 = self._add_noise(x_1, self.sigma_x_1)

            # align x_0 to x_1
            if self.align:
                x_0 = _ase_align(x_0, x_1)

            output = {
                properties.x_0: x_0,
                properties.x_1: x_1,
                f"target_{self.target_property}": transition_state[
                    self.target_property
                ],
                **transition_state,
            }
            output[self.target_property] = output[properties.x_0]

            # starting point is intermedate
            output[properties.image_type] = torch.tensor(
                [properties.IMAGE_TYPES["intermediate"]]
            )

            if self.conditioned:
                # If conditions are present add to input
                output = self._add_conditions(output, [reactant, product])

            outputs = [output]

        return outputs

class DiffusionTS(GenerativeStrategy):
    """
    Diffusion reaction transform. This will prepare the source to be the transition
    state and add reactant and products as condition if wanted.
    """

    def forward(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        # Group inputs by image type
        reactant, product, transition_state, *_ = _analyze_inputs(inputs)

        # Define x_0 for diffusion
        output = {
            properties.x_0: transition_state[self.target_property],
            f"target_{self.target_property}": transition_state[self.target_property],
            **transition_state,
        }
        output[self.target_property] = output[properties.x_0]

        # starting point is noise (intermedate)
        output[properties.image_type] = torch.tensor(
            [properties.IMAGE_TYPES["intermediate"]]
        )

        if self.conditioned:
            # If conditions are present add to input
            output = self._add_conditions(output, [reactant, product])

        outputs = [output]

        return outputs
