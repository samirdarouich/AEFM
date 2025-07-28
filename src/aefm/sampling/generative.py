import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch_scatter import scatter_mean
from torchdiffeq import odeint
from tqdm import tqdm

from aefm import properties
from aefm.processes import DiffusionProcess
from aefm.sampling import Sampler
from aefm.sampling.fixedpoint_solvers import anderson
from aefm.utils import get_repaint_schedule

log = logging.getLogger(__name__)

FIXPOINT_SOLVERS = {
    "anderson": anderson,
}

__all__ = [
    "FixedpointSampler",
    "FlowSampler",
    "DDPMSampler",
]

class FixedpointSampler(Sampler):
    def __init__(
        self,
        property: str,
        property_pred_key: str,
        prediction_net: Union[str, nn.Module],
        fixpoint_algorithm: str = "anderson",
        fixpoint_settings: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(prediction_net=prediction_net, **kwargs)
        self.property = property
        self.property_pred_key = property_pred_key
        self.fixpoint_algorithm = fixpoint_algorithm
        self.fixpoint_settings = fixpoint_settings
        
        log.warning("Batched evaluation is not yet implemented!")

    @torch.no_grad()
    def sample(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Tuple[int, List[Dict[str, torch.Tensor]]]:
        """
        Perform sampling using the flow process. Returns the sampled data for the
        provided time steps.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting flow property.

        Returns:
            Sampled data as list of dictionaries.
        """
        # Check if center of geometry is close to zero
        idx_m = inputs[properties.idx_m]
        if properties.conditions_idx_m in inputs:
            idx_m = inputs[properties.conditions_idx_m]
        CoG = scatter_mean(inputs[properties.R], idx_m, dim=0)
        if self.invariant and (CoG > 1e-5).any():
            raise ValueError(
                "The input positions are not centered, "
                "while the specified diffusion process is invariant."
            )

        nfe, trajectory = self.iterative_update(
            inputs, **self.fixpoint_settings, **kwargs
        )

        # Prepare the output
        batch = {prop: val.clone() for prop, val in inputs.items()}

        if self.results_on_cpu:
            trajectory = trajectory.cpu()  # type: ignore
            batch = {prop: val.cpu() for prop, val in batch.items()}

        trajectory_list = []
        for traj in trajectory:
            batch[self.property] = traj  # type: ignore
            trajectory_list.append(batch.copy())

        return nfe, trajectory_list

    @torch.no_grad()
    def iterative_update(
        self, inputs, **kwargs
    ):
        """
        Iteratively refines x_t towards x_1_pred using adaptive step sizes.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting flow property.
            **kwargs: additional arguments for the fixed point solver.
            
        Returns:
            Number of function evaluations and sampled data along with uncertainty.
        """
        if properties.x_0 in inputs:
            x_0 = inputs[properties.x_0].clone().detach()
        else:
            x_0 = inputs[properties.R].clone().detach()

        # Fixpoint algorithm currently has a batch dimension, which does not work for
        # molecules. So no batching is yet implemented.
        @torch.no_grad()
        def f(x):
            # x has shape (1, n_atoms, 3)
            inputs[self.property] = x.squeeze(0)
            # return shape (1, n_atoms, 3)
            return self.prediction_net(inputs)[self.property_pred_key].unsqueeze(0)
        
        # Adapt the convergence based on the number of atoms (relation norm to RMSD)
        if ("stop_mode" in kwargs) and (kwargs["stop_mode"] == "abs"):
            eps = kwargs["eps"]
            kwargs["eps"] = eps * torch.sqrt(inputs[properties.n_atoms][0])
        
        out = FIXPOINT_SOLVERS[self.fixpoint_algorithm](f, x_0.unsqueeze(0), **kwargs)
        
        # Remove batch dimension of prediction
        trajectory = [traj.squeeze(0) for traj in out["history"]]

        return out["nstep"], torch.stack(trajectory)
    
class FlowSampler(Sampler):
    """
    Implements the plain CFM sampling using torchdiffeq.
    Subclasses the base class 'Sampler'.
    """

    def __init__(
        self,
        flow_property: str,
        velocity_net: Union[str, nn.Module],
        time_key: str = "t",
        velocity_pred_key: str = "vel_pred",
        ode_int_kwargs: Union[Dict[str, Any], None] = None,
        use_vd_ode: bool = False,
        pred_target: bool = False,
        **kwargs,
    ):
        """
        Args:
            flow_property: property to aefm.
            flow_process: the diffusion processe to sample the target property.
            velocity_net: velocity net or path to velocity net to use for the flow
                process.
            time_key: the key for the time.
            velocity_pred_key: the key for the velocity prediction.
        """
        super().__init__(prediction_net=velocity_net, **kwargs)
        self.flow_property = flow_property
        self.time_key = time_key
        self.velocity_pred_key = velocity_pred_key
        self.ode_int_kwargs = ode_int_kwargs or {}
        self.use_vd_ode = use_vd_ode
        self.pred_target = pred_target

    @torch.no_grad()
    def inference_step(
        self,
        inputs: Dict[str, torch.Tensor],
        t: float,
        conditions_in_target: bool = False,
        conditions_zero_velocity: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get the velocity prediction.

        Args:
            inputs: input data for velocity prediction.
            t: the current time of the flow process.
            conditions_in_target: Wheter conditions are also flowed during inference.
            conditions_zero_velocity: Whether conditions have zero velocity.

        Returns:
            velocity prediction and its uncertainty (if available)
        """
        # broadcast the time step to atoms-level
        time_steps = torch.full_like(
            inputs[properties.n_atoms],
            fill_value=t,
            device=self.device,
            dtype=inputs[properties.R].dtype,
        )[inputs[properties.idx_m]]

        # 0 time for conditioning atoms
        if properties.conditions_mask in inputs:
            condition_mask = inputs[properties.conditions_mask] == 0

            # In case jointly model is trained provide correct times for conditions
            if not conditions_in_target:
                time_steps[condition_mask] = 0.0

        # Add time to inputs
        inputs[self.time_key] = time_steps

        # cast input to float for the velocity net
        for key, val in inputs.items():
            if val.dtype == torch.float64:
                inputs[key] = val.float()

        # forward pass through the velocity net
        model_out = self.prediction_net(inputs)  # type: ignore

        # fetch the velocity prediction
        velocity_pred = model_out[self.velocity_pred_key].detach()

        # fetch the uncertainty if available
        velocity_pred_uncertainty = model_out.get(
            self.velocity_pred_key + "_uncertainty", torch.zeros_like(velocity_pred)
        )

        if self.pred_target:
            # NET predicts x_1^ but velocity is v = x_1^-x_0^
            # using x_0^ = (x_t - t*x_1^) / (1-t)
            # v = x_1^ - (x_t - t*x_1^) / (1-t) = (x_1^ - x_t) / (1-t)
            x_t = inputs[self.flow_property]
            velocity_pred = (velocity_pred - x_t) / (1 - time_steps.unsqueeze(1) + 1e-4)

        # 0 velocity for conditioning atoms
        if properties.conditions_mask in inputs and conditions_zero_velocity:
            velocity_pred[condition_mask] = 0.0
            velocity_pred_uncertainty[condition_mask] = 0.0

        return velocity_pred, velocity_pred_uncertainty

    @torch.no_grad()
    def sample(
        self,
        inputs: Dict[str, torch.Tensor],
        t: torch.Tensor,
        conditions_in_target: bool = False,
        t_start: Optional[float] = None,
        **kwargs,
    ) -> Tuple[int, List[Dict[str, torch.Tensor]]]:
        """
        Perform sampling using the flow process. Returns the sampled data for the
        provided time steps.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting flow property.
            t: Time steps for which samples will be saved.

        Returns:
            Sampled data as list of dictionaries.
        """
        # Check if center of geometry is close to zero
        idx_m = inputs[properties.idx_m]
        if properties.conditions_idx_m in inputs:
            idx_m = inputs[properties.conditions_idx_m]
        CoG = scatter_mean(inputs[properties.R], idx_m, dim=0)
        if self.invariant and (CoG > 1e-5).any():
            raise ValueError(
                "The input positions are not centered, "
                "while the specified diffusion process is invariant."
            )

        # Change t to start from t_start
        if t_start is not None:
            t = t.clone()
            t = t[t > t_start]
            t = torch.cat(
                (torch.tensor([t_start], device=t.device, dtype=t.dtype), t), dim=0
            )

        if self.use_vd_ode:
            nfe, trajectory, uncertainty_history = self._sample_vd_ode(
                inputs, t, conditions_in_target, **kwargs
            )
        else:
            nfe, trajectory, uncertainty_history = self._sample(
                inputs, t, conditions_in_target, **kwargs
            )

        # Prepare the output
        batch = {prop: val.clone() for prop, val in inputs.items()}

        if self.results_on_cpu:
            trajectory = trajectory.cpu()  # type: ignore
            batch = {prop: val.cpu() for prop, val in batch.items()}

        trajectory_list = []
        for time_step, traj, uncertainty in zip(t, trajectory, uncertainty_history):
            batch[self.flow_property] = traj  # type: ignore
            batch[self.flow_property + "_uncertainty"] = uncertainty
            batch[self.time_key] = time_step.repeat(batch[properties.n_atoms].sum())
            trajectory_list.append(batch.copy())

        return nfe, trajectory_list

    @torch.no_grad()
    def _sample(
        self,
        inputs: Dict[str, torch.Tensor],
        t: torch.Tensor,
        conditions_in_target: bool = False,
        **kwargs,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Perform sampling using the flow process. Returns the sampled data for the
        provided time steps.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting flow property.
            t: Time steps for which samples will be saved.

        Returns:
            Number of function evaluations and sampled data.
        """
        if t.max() > 1.0 or t.min() < 0.0:
            raise ValueError("t must be between 0 and 1.")

        # Copy inputs to avoid inplace operations
        batch = {prop: val.clone().to(self.device) for prop, val in inputs.items()}
        t = t.clone().to(self.device)
        self.nfe = 0

        # Get initial value
        y0 = batch[self.flow_property]
        if properties.x_0 in batch:
            if properties.conditions_mask in batch:
                if conditions_in_target:
                    # In case of jointly training structure and conditons
                    y0 = batch[properties.x_0]
                else:
                    # In case of only training the flow for one structure and pass the
                    # others as conditions
                    condition_mask = batch[properties.conditions_mask] == 1
                    y0[condition_mask] = batch[properties.x_0]
            else:
                y0 = batch[properties.x_0]

        def ode_func(t, y):
            """
            ODE function wrapper.
            """
            # Increase counter
            self.nfe += 1

            # Update flow property with current state
            batch[self.flow_property] = y

            # Get velocity prediction
            vel, vel_uncertainty = self.inference_step(
                batch, t, conditions_in_target=conditions_in_target
            )

            return vel

        # Solve the ODE
        trajectory = odeint(ode_func, y0, t, **self.ode_int_kwargs)
        #! Uncertainty is not implemented in the current implementation
        uncertainty_history = torch.zeros_like(trajectory)

        return self.nfe, trajectory, uncertainty_history

    @torch.no_grad()
    def _sample_vd_ode(
        self,
        inputs: Dict[str, torch.Tensor],
        t: torch.Tensor,
        conditions_in_target: bool = False,
        **kwargs,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Perform sampling using the flow process. Returns the sampled data for the
        provided time steps.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting flow property.
            t: Time steps for which samples will be saved.

        Returns:
            Number of function evaluations and sampled data along with uncertainty.
        """
        if t.max() > 1.0 or t.min() < 0.0:
            raise ValueError("t must be between 0 and 1.")

        # Copy inputs to avoid inplace operations
        batch = {prop: val.clone().to(self.device) for prop, val in inputs.items()}
        t_save = t.clone().to(self.device)

        # Get initial value
        y0 = batch[self.flow_property]
        if properties.x_0 in batch:
            if properties.conditions_mask in batch:
                if conditions_in_target:
                    # In case of jointly training structure and conditons
                    y0 = batch[properties.x_0]
                else:
                    # In case of only training the flow for one structure and pass the
                    # others as conditions
                    condition_mask = batch[properties.conditions_mask] == 1
                    y0[condition_mask] = batch[properties.x_0]
            else:
                y0 = batch[properties.x_0]

        # Solve the Variance Diminishing ODE
        clamp_min = torch.tensor(1e-6, device=self.device, dtype=y0.dtype)
        clamp_max = torch.tensor(1 - 1e-6, device=self.device, dtype=y0.dtype)
        self.nfe = 0
        trajectory = []
        uncertainty_history = []
        batch[self.flow_property] = y0
        i = len(t_save) - 1
        for n in range(i):
            t_ = t_save[n]
            s_ = t_save[n + 1]

            vel, vel_uncertainty = self.inference_step(
                batch,
                t_,
                conditions_in_target=conditions_in_target,
                conditions_zero_velocity=False,
            )

            # Set initial uncertainty
            if n == 0:
                uncertainty = vel_uncertainty

            if t_ in t_save:
                trajectory.append(batch[self.flow_property].clone().to(y0.dtype))
                uncertainty_history.append(uncertainty.clone().to(y0.dtype))

            # Set prediction to known values for conditioned atoms
            if properties.conditions_mask in batch:
                condition_mask = batch[properties.conditions_mask] == 0
                vel[condition_mask] = y0[condition_mask].to(dtype=vel.dtype)
                vel_uncertainty[condition_mask] = 0.0

            current_weight = torch.clamp(
                (1 - s_) / (1 - t_),
                min=clamp_min,
                max=clamp_max,
            )
            vel_weight = torch.clamp(
                1 - (1 - s_) / (1 - t_),
                min=clamp_min,
                max=clamp_max,
            )

            # Update flow property with current state
            batch[self.flow_property] = (
                batch[self.flow_property] * current_weight + vel * vel_weight
            )

            # Update uncertainty assuming no corelation between velocity and uncertainty
            uncertainty = torch.sqrt(
                torch.pow(uncertainty * current_weight, 2)
                + torch.pow(vel_uncertainty * vel_weight, 2)
            )

            self.nfe += 1

        # Add final structure
        trajectory.append(batch[self.flow_property].clone().to(y0.dtype))
        trajectory = torch.stack(trajectory)
        uncertainty_history.append(uncertainty.clone().to(y0.dtype))
        uncertainty_history = torch.stack(uncertainty_history)
        assert len(trajectory) == len(t_save), "Always provide t=1.0 as save point"

        return self.nfe, trajectory, uncertainty_history


class DDPMSampler(Sampler):
    def __init__(
        self,
        diffuse_property: str,
        denoiser_net: Union[str, nn.Module],
        diffusion_process: DiffusionProcess,
        time_key: str = "t",
        noise_pred_key: str = "eps_pred",
        resamplings: Optional[int] = None,
        jump_length: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            diffusion_process: The diffusion processe to sample the target property.
            denoiser: denoiser or path to denoiser to use for the reverse process.
            time_key: the key for the time.
            noise_pred_key: the key for the noise prediction.
        """
        super().__init__(prediction_net=denoiser_net, **kwargs)
        self.diffuse_property = diffuse_property
        self.diffusion_process = diffusion_process
        self.time_key = time_key
        self.noise_pred_key = noise_pred_key

        self.inpaint_flag = False
        if resamplings is not None and jump_length is not None:
            self.resamplings = resamplings
            self.jump_length = jump_length
            self.schedule = get_repaint_schedule(
                self.diffusion_process.get_T(), jump_length, resamplings
            )
            self.inpaint_flag = True

    @torch.no_grad()
    def inference_step(
        self,
        inputs: Dict[str, torch.Tensor],
        iter: int,
        conditions_in_target: bool = False,
        conditions_zero_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get the noise prediction.

        Args:
            inputs: input data for noise prediction.
            t: the current time of the flow process.
            conditions_in_target: Wheter conditions are also diffused during inference.
            conditions_zero_noise: Whether conditions have zero noise.

        Returns:
            noise prediction
        """
        # current reverse time step
        time_steps = torch.full_like(
            inputs[properties.n_atoms],
            fill_value=iter,
            dtype=torch.long,
            device=self.device,
        )

        # append the normalized time step to the model input
        inputs[self.time_key] = self.diffusion_process.normalize_time(time_steps)

        # broadcast the time step to atoms-level
        inputs[self.time_key] = inputs[self.time_key][inputs[properties.idx_m]]

        # 0 time for conditioning atoms
        if properties.conditions_mask in inputs:
            condition_mask = inputs[properties.conditions_mask] == 0

            # In case jointly model is trained provide correct times for conditions
            if not conditions_in_target:
                inputs[self.time_key][condition_mask] = 0.0

        # cast input to float for the denoiser
        for key, val in inputs.items():
            if val.dtype == torch.float64:
                inputs[key] = val.float()

        # forward pass through the denoiser
        model_out = self.prediction_net(inputs)  # type: ignore

        # fetch the noise prediction
        noise_pred = model_out[self.noise_pred_key].detach()

        # 0 noise for conditioning atoms
        if properties.conditions_mask in inputs and conditions_zero_noise:
            noise_pred[condition_mask] = 0.0

        return time_steps, noise_pred

    @torch.no_grad()
    def sample(
        self,
        inputs: Dict[str, torch.Tensor],
        t: torch.Tensor,
        t_start: Optional[int] = None,
        conditions_in_target: bool = False,
        **kwargs,
    ) -> Tuple[int, List[Dict[str, torch.Tensor]]]:
        """
        Peforns denoising/sampling using the reverse diffusion process.
        Returns the denoised/sampled data, the number of steps taken,
        and the progress history if saved.

        Either performs pure (conditioned) generation or inpainting based on the flag.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting x_t.
            t: Time steps for which samples will be saved (between 0 and 1)
            conditions_in_target: Wheter conditions are also diffused during inference.
        """

        # Check if center of geometry is close to zero
        idx_m = inputs[properties.idx_m]
        if properties.conditions_idx_m in inputs:
            idx_m = inputs[properties.conditions_idx_m]
        CoG = scatter_mean(inputs[properties.R], idx_m, dim=0)
        if self.invariant and (CoG > 1e-5).any():
            raise ValueError(
                "The input positions are not centered, "
                "while the specified diffusion process is invariant."
            )

        if self.inpaint_flag:
            assert conditions_in_target, "Inpainting requires conditions in target"
            return self._sample_inpaint(
                inputs, t, t_start, conditions_in_target, **kwargs
            )
        else:
            return self._sample(inputs, t, t_start, conditions_in_target, **kwargs)

    @torch.no_grad()
    def _sample(
        self,
        inputs: Dict[str, torch.Tensor],
        t: torch.Tensor,
        t_start: Optional[int] = None,
        conditions_in_target: bool = False,
        **kwargs,
    ) -> Tuple[int, List[Dict[str, torch.Tensor]]]:
        """
        Peforms denoising/sampling using the reverse diffusion process.
        Returns the denoised/sampled data, the number of steps taken,
        and the progress history if saved.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting x_t.
            t: Time steps for which samples will be saved (between 0 and 1)
            conditions_in_target: Wheter conditions are also diffused during inference.
        """
        
        # Default is t=T
        if t_start is None:
            sample_prior = True
            t_start = self.diffusion_process.get_T()
        else:
            sample_prior = False

        # Unnormalize the time steps
        t_save = self.diffusion_process.unnormalize_time(t)

        # copy inputs to avoid inplace operations
        batch = {prop: val.clone().to(self.device) for prop, val in inputs.items()}

        # history of the reverse steps
        trajectory = []

        # Get prior (only for unconditioned atoms)
        if sample_prior:
            if properties.conditions_mask in batch:
                condition_mask = batch[properties.conditions_mask] == 1
                x0 = batch[self.diffuse_property]
                x0[condition_mask] = self.diffusion_process.sample_prior(
                    batch[self.diffuse_property][condition_mask],
                    batch[properties.idx_m][condition_mask],
                ).to(dtype=x0.dtype)
            else:
                x0 = self.diffusion_process.sample_prior(
                    batch[self.diffuse_property], batch[properties.idx_m]
                ).to(dtype=batch[self.diffuse_property].dtype)

            batch[self.diffuse_property] = x0

        # simulate the reverse process
        nfe = 0
        for i in tqdm(range(t_start - 1, -1, -1)):
            # get the time steps and noise predictions from the denoiser
            time_steps, noise = self.inference_step(batch, i, conditions_in_target)

            # save history if required. Must be done before the reverse step.
            if i in t_save and not i == 0:
                trajectory.append(batch[self.diffuse_property].cpu().clone())

            # perform one reverse step (only on unconditioned atoms)
            if properties.conditions_mask in batch:
                new_property = self.diffusion_process.reverse_step(
                    batch[self.diffuse_property][condition_mask],
                    noise[condition_mask],
                    batch[properties.idx_m][condition_mask],
                    time_steps[inputs[properties.idx_m][condition_mask]],
                )

                batch[self.diffuse_property][condition_mask] = new_property

            else:
                batch[self.diffuse_property] = self.diffusion_process.reverse_step(
                    batch[self.diffuse_property],
                    noise,
                    batch[properties.idx_m],
                    time_steps[inputs[properties.idx_m]],
                )

            nfe += 1

        # Add final structure
        trajectory.append(batch[self.diffuse_property].clone())
        assert len(trajectory) == len(t_save), "Always provide t=1.0 as save point"

        # Prepare the output
        if self.results_on_cpu:
            trajectory[-1] = trajectory[-1].cpu()
            trajectory = torch.stack(trajectory)
            batch = {prop: val.clone().cpu() for prop, val in inputs.items()}

        trajectory_list = []
        for time_step, traj in zip(t_save.flip(0), trajectory):
            batch[self.diffuse_property] = traj  # type: ignore
            batch[self.time_key] = time_step
            trajectory_list.append(batch.copy())

        return nfe, trajectory_list

    @torch.no_grad()
    def _sample_inpaint(
        self,
        inputs: Dict[str, torch.Tensor],
        t: torch.Tensor,
        t_start: Optional[int] = None,
        conditions_in_target: bool = True,
        **kwargs,
    ) -> Tuple[int, List[Dict[str, torch.Tensor]]]:
        """
        Peforms denoising/sampling using the reverse diffusion process and inpainting.
        Returns the denoised/sampled data, the number of steps taken, and the progress
        history if saved.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting x_t.
            t: Time steps for which samples will be saved (between 0 and 1)
            conditions_in_target: Wheter conditions are also diffused during inference.
        """
        assert conditions_in_target, "Inpainting requires conditions in target"
        # Unnormalize the time steps
        t_save = self.diffusion_process.unnormalize_time(t)

        # copy inputs to avoid inplace operations
        batch = {prop: val.clone().to(self.device) for prop, val in inputs.items()}

        # history of the reverse steps
        trajectory = []
        seen_ts = []

        known_mask = batch[properties.conditions_mask] == 0
        unknown_mask = batch[properties.conditions_mask] == 1
        idx_m = batch[properties.conditions_idx_m]

        # Save known parts and get prior
        x0 = batch[properties.x_0]
        batch[self.diffuse_property] = self.diffusion_process.sample_prior(
            batch[self.diffuse_property], idx_m
        ).to(dtype=batch[self.diffuse_property].dtype)

        # simulate the reverse process using repaint
        nfe = 0
        for t_last, t_cur in zip(self.schedule[:-1], self.schedule[1:]):
            # Reverse step
            if t_cur < t_last:
                # get the time steps and noise predictions from the denoiser
                time_steps, noise = self.inference_step(
                    batch, t_last, conditions_in_target, conditions_zero_noise=False
                )

                # save history if required. Must be done before the reverse step.
                # Avoid saving duplicates at same time step
                if t_last in t_save and not t_last == 0 and t_last not in seen_ts:
                    trajectory.append(batch[self.diffuse_property].cpu().clone())
                    seen_ts.append(t_last)

                # Get known parts
                known, _ = self.diffusion_process.diffuse(
                    x0,
                    idx_m=idx_m,
                    t=time_steps[batch[properties.idx_m]],
                )

                # Get unknown parts
                new_property = self.diffusion_process.reverse_step(
                    batch[self.diffuse_property],
                    noise,
                    idx_m,
                    time_steps[batch[properties.idx_m]],
                )

                # Combine known and unknown
                batch[self.diffuse_property] = known * known_mask.unsqueeze(
                    1
                ) + new_property * unknown_mask.unsqueeze(1)

            # Forward step
            else:
                time_steps = torch.full_like(
                    batch[properties.n_atoms],
                    fill_value=t_last,
                    dtype=torch.long,
                )
                batch[self.diffuse_property], _ = self.diffusion_process.forward_step(
                    batch[self.diffuse_property],
                    idx_m=idx_m,
                    t_next=time_steps[batch[properties.idx_m]],
                )

            nfe += 1

        # Add final structure
        trajectory.append(batch[self.diffuse_property].clone())
        assert len(trajectory) == len(t_save), "Always provide t=1.0 as save point"

        # Prepare the output
        if self.results_on_cpu:
            trajectory[-1] = trajectory[-1].cpu()
            trajectory = torch.stack(trajectory)
            batch = {prop: val.clone().cpu() for prop, val in inputs.items()}

        trajectory_list = []
        for time_step, traj in zip(t_save.flip(0), trajectory):
            batch[self.diffuse_property] = traj  # type: ignore
            batch[self.time_key] = time_step
            trajectory_list.append(batch.copy())

        return nfe, trajectory_list


