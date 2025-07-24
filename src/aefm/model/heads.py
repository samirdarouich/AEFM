from typing import Callable, Dict, Optional, Sequence, Union

import schnetpack as spk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from aefm import properties

__all__ = [
    "TimePrediction",
    "TimeAwareAtomwise",
    "TimeAwareEquivariant",
]


class TimePrediction(nn.Module):
    """
    Invariant time prediction head using the spk.atomistic.Atomwise module.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = "t_pred",
        aggregation_mode: Optional[str] = None,
        detach_representation: bool = False,
    ):
        """
        Args:
            n_in: input dimension without time.
            n_out: time output dimension.
            n_hidden: size of hidden layers.
            n_layers: number of hidden layers.
            activation: activation function.
            output_key: the key under which the result will be stored.
            aggregation_mode: aggregation mode for the atomwise module.
            detach_representation: detachs representation module
                                from computational graph.
        """
        super().__init__()

        self.n_out = n_out

        self.aggregation_mode = aggregation_mode
        self.output_key = output_key
        self.model_outputs = [output_key]

        # time predictor MLP
        self.time_outnet = spk.atomistic.Atomwise(
            n_in=n_in,
            n_out=self.n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            aggregation_mode=aggregation_mode,
            output_key=self.output_key,
            per_atom_output_key=self.output_key,
        )

        self.detach_representation = detach_representation

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # stop gradient flow through backbone representation.
        if self.detach_representation:
            det_inputs = {key: val.detach() for key, val in inputs.items()}
            det_inputs = self.time_outnet(det_inputs)
            inputs[self.output_key] = det_inputs[self.output_key]

        else:
            inputs = self.time_outnet(inputs)

        if self.aggregation_mode is None:
            inputs[self.output_key] = inputs[self.output_key].squeeze(-1)

        return inputs


class TimeAwareAtomwise(spk.atomistic.Atomwise):
    """
    Time-aware Atomwise head for energy prediction.
    Overwrites the ``spk.atomistic.Atomwise`` module.
    Can be used to predict diffusion noise as the forces, i.e. derivative of energy.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        include_time: bool = False,
        time_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            n_in: input dimension without time.
            n_out: output dimension for the target property.
            include_time: whether to append time as input feature.
            time_key: time key for input to the Atomwise module.
        """

        # append time as input feature
        self.include_time = include_time
        if self.include_time:
            n_in += 1

        super().__init__(n_in, n_out=n_out, **kwargs)

        self.time_key = time_key
        # set default time key
        if self.include_time and self.time_key is None:
            raise ValueError(
                "Argument 'time_key' must be set when 'include_time' is True."
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.include_time:
            t = inputs[self.time_key]  # type: ignore
            t = t.unsqueeze(-1)

            # append time to scalar representation features
            inputs["scalar_representation"] = torch.cat(
                (inputs["scalar_representation"], t), dim=-1
            )

        # predict atomwise contributions
        inputs = super().forward(inputs)

        return inputs


class TimeAwareEquivariant(nn.Module):
    """
    Time-aware Equivariant head for conditional velocity.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        include_time: bool = False,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = "v_t_pred",
        time_head: Optional[nn.Module] = None,
        time_key: Optional[str] = None,
        detach_time: bool = False,
    ):
        """
        Args:
            n_in: input dimension without time.
            n_out: output dimension for the target property.
            include_time: whether to append time as input feature.
            n_hidden: size of hidden layers.
                    If an integer, same number of node is used for all hidden
                        layers resulting in a rectangular network.
                    If None, the number of neurons is divided
                        by two after each layer starting
                        n_in resulting in a pyramidal network.
            n_layers: number of hidden layers.
            activation: activation function.
            output_key: the key under which the result will be stored.
            time_head: time head to predict time.
            time_key: time key for input to the equivariant module.
            detach_time: detach time prediction from computational graph.
        """

        super().__init__()

        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )

        self.output_key = output_key
        self.model_outputs = [output_key]

        # Add uncertainty in case of shallow ensemble
        if n_out > 1:
            self.model_outputs.append(self.output_key + "_uncertainty")

        self.include_time = include_time
        # add time as input scalar feature
        if self.include_time:
            self.outnet[0] = spk.nn.GatedEquivariantBlock(
                n_sin=self.outnet[0].n_sin + 1,
                n_vin=self.outnet[0].n_vin,
                n_sout=self.outnet[0].n_sout,
                n_vout=self.outnet[0].n_vout,
                n_hidden=self.outnet[0].n_hidden,
                activation=activation,
                sactivation=activation,
            )

        # time prediction
        self.time_outnet = time_head
        self.detach_time = detach_time

        self.time_key = time_key
        if self.include_time and self.time_key is None:
            raise ValueError(
                "Argument 'time_key' must be set when 'include_time' is True."
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]

        # append time to representation
        if self.include_time:
            # predict time
            if self.time_outnet is not None:
                t = self.time_outnet(inputs)[self.time_key]
                if self.time_outnet.aggregation_mode is not None:
                    t = t[inputs[properties.idx_m]]
            # use true time from input
            else:
                t = inputs[self.time_key]  # type: ignore

            # detach time from computational graph
            if self.detach_time:
                t = t.detach()

            t = t.unsqueeze(-1)

            # append time to scalar representation features
            l0 = torch.cat((l0, t), dim=-1)

        # predict equivariant output
        _, x = self.outnet((l0, l1))

        # Get uncertainty
        if x.shape[-1] > 1:

            idx_m = inputs[properties.idx_m]
            if properties.conditions_idx_m in inputs:
                idx_m = inputs[properties.conditions_idx_m]

            # cog of each prediction
            cog = scatter_mean(x, idx_m.unsqueeze(-1), dim=0)[idx_m]
            x = x - cog

            # get std of each prediction
            std = x.std(dim=-1)

            # get mean of the predictions
            x = x.mean(dim=-1)

            inputs[self.output_key] = x
            inputs[self.output_key + "_uncertainty"] = std
        else:
            x = torch.squeeze(x, -1)
            inputs[self.output_key] = x

        return inputs