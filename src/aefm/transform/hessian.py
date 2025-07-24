import logging
from copy import deepcopy
from typing import Callable, Dict, Tuple, Union

import ase.units as units
import numpy as np
import schnetpack
import schnetpack.transform as trn
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import atomic_masses
from ase.vibrations import Vibrations
from schnetpack.utils import load_model
from torch import nn

from aefm import properties
from aefm.utils.analysis import inputs_to_atoms

log = logging.getLogger(__name__)

__all__ = [
    "PhysicalVelocity",
    "NormalModeSampling",
]


def _calculate_translational_rotational_projection(
    mass: torch.Tensor, coords: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Projection matrix to project out translational and rotational modes
    for a given set of masses and coordinates.

    Compare:
    https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/thermo.py

    Args:
        mass: The atomic masses in units of amu. Shape is (N,).
        coords: The atomic positions in units of Å. Shape is (N, 3).

    Returns:
        Projection matrix P.

    Procedure:
        1. Calculate the center of mass of the system.
        2. Translate the coordinates to the center of mass frame.
        3. Compute the translational modes (Tx, Ty, Tz).
        4. Calculate the moment of inertia tensor and its eigenvalues and eigenvectors.
        5. Determine the principal axes of rotation.
        6. Compute the rotational modes (Rx, Ry, Rz) in the principal axes frame.
    """
    mass_center = torch.einsum("z,zx->x", mass, coords) / mass.sum()
    coords = coords - mass_center
    massp = mass**0.5

    # translational mode
    Tx = torch.einsum(
        "m,x->mx", massp, torch.tensor([1, 0, 0], dtype=mass.dtype, device=mass.device)
    )
    Ty = torch.einsum(
        "m,x->mx", massp, torch.tensor([0, 1, 0], dtype=mass.dtype, device=mass.device)
    )
    Tz = torch.einsum(
        "m,x->mx", massp, torch.tensor([0, 0, 1], dtype=mass.dtype, device=mass.device)
    )

    im = torch.einsum("m,mx,my->xy", mass, coords, coords)
    im = torch.eye(3, dtype=mass.dtype, device=mass.device) * im.trace() - im
    w, paxes = torch.linalg.eigh(im)

    # make the z-axis be the rotation vector with the smallest moment of inertia
    w = w.flip(0)
    paxes = paxes.flip(1)
    ex, ey, ez = paxes.T

    coords_in_rot_frame = coords @ paxes
    cx, cy, cz = coords_in_rot_frame.T
    Rx = massp[:, None] * (cy[:, None] * ez - cz[:, None] * ey)
    Ry = massp[:, None] * (cz[:, None] * ex - cx[:, None] * ez)
    Rz = massp[:, None] * (cx[:, None] * ey - cy[:, None] * ex)

    # Get projection matrix
    TRspace = torch.vstack(
        (
            Tx.flatten(),
            Ty.flatten(),
            Tz.flatten(),
            Rx.flatten(),
            Ry.flatten(),
            Rz.flatten(),
        )
    )

    q, r = torch.linalg.qr(TRspace.T)
    P = torch.eye(mass.shape[0] * 3, dtype=mass.dtype, device=mass.device) - q @ q.T

    return P


def _reshape_hessian(N: int, hessian: torch.Tensor) -> torch.Tensor:
    """Reshape the Hessian matrix to 3N x 3N."""
    if hessian.shape == (N * N, 3, 3):
        # Reshape the hessian into (N, N, 3, 3)
        hessian_reshaped = hessian.reshape(N, N, 3, 3)

        # Rearrange into (3N, 3N) Hessian
        rows = [
            torch.cat([hessian_reshaped[i, j] for j in range(N)], dim=1)
            for i in range(N)
        ]
        hessian = torch.cat(rows, dim=0)
    else:
        pass

    assert hessian.shape == (
        3 * N,
        3 * N,
    ), f"Expected Hessian to have shape `(3N, 3N)`, got {hessian.shape}"
    return hessian


def _remove_condition_nodes(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove conditioning nodes from input.
    """
    outputs = deepcopy(inputs)
    if properties.conditions_mask in inputs:
        t_mask = inputs[properties.conditions_mask].nonzero()[:, 0]

        # Remove condition nodes
        outputs[properties.R] = inputs[properties.R][t_mask]
        outputs[properties.Z] = inputs[properties.Z][t_mask]
        outputs[properties.n_atoms] = torch.full_like(
            inputs[properties.n_atoms], len(t_mask)
        )

        if properties.idx_m in inputs:
            outputs[properties.idx_m] = inputs[properties.idx_m][t_mask]
        else:
            outputs[properties.idx_m] = torch.repeat_interleave(
                torch.arange(1), repeats=outputs[properties.n_atoms], dim=0
            )

        # Remove all conditioning sending nodes
        if properties.idx_i in inputs:
            ij_mask = torch.isin(inputs[properties.idx_i], t_mask) & torch.isin(
                inputs[properties.idx_j], t_mask
            )
            outputs[properties.idx_i] = inputs[properties.idx_i][ij_mask]
            outputs[properties.idx_j] = inputs[properties.idx_j][ij_mask]
            outputs[properties.offsets] = inputs[properties.offsets][ij_mask]

        if properties.idx_i_triples in inputs:
            ijk_mask = (
                torch.isin(inputs[properties.idx_i_triples], t_mask)
                & torch.isin(inputs[properties.idx_j_triples], t_mask)
                & torch.isin(inputs[properties.idx_k_triples], t_mask)
            )
            outputs[properties.idx_i_triples] = inputs[properties.idx_i_triples][
                ijk_mask
            ]
            outputs[properties.idx_j_triples] = inputs[properties.idx_j_triples][
                ijk_mask
            ]
            outputs[properties.idx_k_triples] = inputs[properties.idx_k_triples][
                ijk_mask
            ]

    return outputs


def _in_memory_hessian(
    atoms: Atoms, **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Hessian matrix and equilibrium forces for a given set of atoms.

    This function uses the Vibrations class to displace the atoms and calculate the
    Hessian matrix and equilibrium forces in memory.

    Args:
        atoms: The atomic configuration for which the Hessian matrix and
            equilibrium forces are to be calculated.
        **kwargs: Additional keyword arguments to be passed to the Vibrations class.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the Hessian matrix and
        the equilibrium forces.
    """
    energy_eq = atoms.get_potential_energy()

    vib = Vibrations(atoms, **kwargs)

    cache = {}
    for disp, atoms in vib.iterdisplace(inplace=True):
        result = vib.calculate(atoms, disp)

        cache[disp.name] = result

    vib.cache = cache
    vib.get_frequencies()

    hessian = vib.H
    f_eq = vib._eq_disp().forces()

    assert hessian is not None and hessian.shape == (
        3 * len(atoms),
        3 * len(atoms),
    ), "Expected Hessian to have shape `(3N, 3N)`"

    return hessian, f_eq, energy_eq


class RationalFunctionOptimization:
    def __init__(self, H: torch.Tensor, g: torch.Tensor, order: int) -> None:
        """
        Class for optimizing a system based on rational function optimization (RFO).

        This class performs rational function optimization on a quadratic model of the
        system, using the Hessian matrix and gradient to calculate optimization steps.

        Args:
            H: The Hessian matrix of the system, with shape (3N, 3N).
            g: The gradient vector of the system, with shape (3N,).
            order: Specifies the optimization order. Typically, `0` for minimization
                (other modes) and `1` for transition state maximization.
        """
        self.H = H
        self.g = g
        self.order = order
        self.A = torch.block_diag(
            self.H, torch.tensor([[0]], dtype=H.dtype, device=H.device)
        )
        self.A[-1, :-1] = self.g
        self.A[:-1, -1] = self.g

    def get_s(self) -> torch.Tensor:
        """
        Solves for the optimization step `s` by computing the eigenvalues and
        eigenvectors of the augmented matrix A.

        The step vector `s` is computed based on the `order` (minimization or
        maximization) by solving the rational function optimization problem
        (referencing Baker 1986, Equation 6).

        Returns:
            torch.Tensor:
              The optimization step vector `s` with shape (3N,).
        """
        L, V = torch.linalg.eigh(self.A)
        s = V[:-1, self.order] / V[-1, self.order]

        if self.order == 0:
            log.debug(f"Lambda for minimising other modes: {L[0]:.4f}")
        elif self.order == 1:
            log.debug(f"Lambda for maximising TS mode: {L[-1]:.4f}")

        return s


def _prfo_step(
    g: torch.Tensor, V: torch.Tensor, eigenvals: torch.Tensor
) -> torch.Tensor:
    """
    Perform a quasi-Newton optimization step using the Partitioned Rational Function
    Optimization (p-RFO) method (https://doi.org/10.1002/jcc.540070402).

    This function projects the gradient onto the normal modes, calculates the appropriate
    eigenvalue shifts (lambdas), and updates the atomic positions using the p-RFO step.

    Check this implementation:
    https://github.com/zadorlab/sella/blob/master/sella/optimize/stepper.py

    Args:
        g: The gradients acting on the atoms in units of eV/Å/sqrt(amu). Shape is (3*N).
        V: The eigenvector in mass system in units of eV/Å**2/sqrt(amu). Shape is
            (3N,3N-6).
        eigenvals: The eigenvalues in mass system in units of eV/Å**2/sqrt(amu). Shape
            is (3N-6).

    Returns:
        torch.Tensor:
          The update step for atomic positions in mass system, of shape `(N, 3)`, where
          each entry represents the x, y, z displacement of the respective atom. Unit is
          Angstrom.
    """
    # Indices of the transition vector (which forms the small RFO partition) and the
    # other vectors.
    tv = [0]
    ot = np.array([i for i in range(V.shape[1]) if i not in tv])

    # Get eigenvectors (eV/(A^2*amu)) for following up- / downwards.
    Vmax = V[:, tv]
    Vmin = V[:, ot]

    # Solve p-RFO following one mode upwards and all other modes downwards
    # use V.T with units A^2*amu/eV.
    # Units of V.T @ g: A^2*amu/eV * eV/(A*sqrt(amu)) = A*sqrt(amu)
    rfo_max = RationalFunctionOptimization(
        g=Vmax.T @ g, H=torch.diag(eigenvals[tv]), order=Vmax.shape[1]
    )
    rfo_min = RationalFunctionOptimization(
        g=Vmin.T @ g, H=torch.diag(eigenvals[ot]), order=0
    )

    # Units of s = A*sqrt(amu)/(ev/(A^2*amu))
    smax = rfo_max.get_s()
    smin = rfo_min.get_s()

    # Units of dx = A*sqrt(amu)
    dx = Vmax @ smax + Vmin @ smin

    return dx


def prfo_step_with_tr_projection(
    hessian: torch.Tensor,
    forces: torch.Tensor,
    mass: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """
    Perform a quasi-Newton optimization step using the Partitioned Rational Function
    Optimization (p-RFO) method (https://doi.org/10.1002/jcc.540070402).

    This function projects the gradient onto the normal modes, calculates the appropriate
    eigenvalue shifts (lambdas), and updates the atomic positions using the p-RFO step.

    Check this implementation:
    https://github.com/zadorlab/sella/blob/master/sella/optimize/stepper.py

    The Hessian is converted in mass system and a projection matrix is used to project
    out the translational and rotational modes. The eigenvalues with amount lower than
    1e-8 are identified as translations and rotations.

    Args:
        H: The Hessian matrix in units of eV/Å², representing the second derivatives of
            the potential energy with respect to atomic positions. Shape is (3N, 3N).
        forces: The forces acting on the atoms in units of eV/Å. Shape is (N, 3).
        mass: The atomic masses in units of amu. Shape is (N,).
        coords: The atomic positions in units of Å. Shape is (N, 3).

    Returns:
        torch.Tensor:
          The update step for atomic positions in mass system, of shape `(N, 3)`, where
          each entry represents the x, y, z displacement of the respective atom. Unit is
          Angstrom.
    """
    # Get projection matrix
    P = _calculate_translational_rotational_projection(mass, coords)

    # Mass weight forces (ev/(A*sqrt(amu)) and Hessian (ev/(A**2*sqrt(amu))
    sqrtmmm = mass.repeat_interleave(3).sqrt()
    sqrtmmminv = 1.0 / sqrtmmm
    H = P.T @ torch.einsum("i,ij,j->ij", sqrtmmminv, hessian, sqrtmmminv) @ P
    g_mass = -P.T @ (forces.flatten() * sqrtmmminv)

    # Get eigenvalues shape(dof) and eigenvectors shape(dof,modes) of Hessian
    # unit: eV/A^2/amu
    omega, V = torch.linalg.eigh(H)

    # Sort out modes and eigenvalues that belong to translations and rotations
    idx_nm = torch.ones(len(omega), dtype=torch.bool, device=omega.device)
    idx_nm[omega.abs().argsort()[:6]] = False

    if (omega.shape[0] - (omega.abs() > 1e-4).sum()) != 6:
        log.warning(
            "Other than six small modes (translation and rotation) found. "
            "Smallest seven eigenvalues (eV/A^2/amu): %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f"
            % tuple(omega.abs().sort()[0][:7])
        )

    # Get eigenvalues (unit: eV/A^2/amu)
    eigenvals = omega[idx_nm]

    log.debug(
        "Eigenvalues (eV/A^2/amu): %.3f, %.3f, %.3f, .... %.3f, %.3f, %.3f"
        % (*eigenvals[:3], *eigenvals[-3:])
    )

    # Filter out unwanted modes (row equals normal mode, unit: eV/A^2/amu)
    V = V[:, idx_nm]

    # Get step (unit: A*sqrt(amu))
    step = _prfo_step(g_mass, V, eigenvals)

    # Project back and multiply with inverse square mass to transform units to A
    step = (P @ step * sqrtmmminv).reshape(-1, 3)

    return step


def prfo_step(
    hessian: torch.Tensor,
    forces: torch.Tensor,
    mass: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """
    Perform a quasi-Newton optimization step using the Partitioned Rational Function
    Optimization (p-RFO) method (https://doi.org/10.1002/jcc.540070402).

    This function projects the gradient onto the normal modes, calculates the appropriate
    eigenvalue shifts (lambdas), and updates the atomic positions using the p-RFO step.

    Check this implementation:
    https://github.com/zadorlab/sella/blob/master/sella/optimize/stepper.py

    The Hessian is converted in mass system and the smallest in amount eigenvalues are
    filtered out.

    Args:
        H: The Hessian matrix in units of eV/Å², representing the second derivatives of
            the potential energy with respect to atomic positions. Shape is (3N, 3N).
        forces: The forces acting on the atoms in units of eV/Å. Shape is (N, 3).
        mass: The atomic masses in units of amu. Shape is (N,).
        coords: The atomic positions in units of Å. Shape is (N, 3).

    Returns:
        torch.Tensor:
          The update step for atomic positions in mass system, of shape `(N, 3)`, where
          each entry represents the x, y, z displacement of the respective atom. Unit is
          Angstrom.
    """
    # Mass weight forces and Hessian
    sqrtmmm = mass.repeat_interleave(3).sqrt()
    sqrtmmminv = 1.0 / sqrtmmm
    H = torch.einsum("i,ij,j->ij", sqrtmmminv, hessian, sqrtmmminv)
    g_mass = -forces.flatten() * sqrtmmminv

    # Get eigenvalues shape(dof) and eigenvectors shape(dof,modes) of Hessian
    # unit: eV/A^2/amu
    omega, V = torch.linalg.eigh(H)

    # Sort out modes and eigenvalues that belong to translations and rotations
    idx_nm = torch.ones(len(omega), dtype=torch.bool, device=omega.device)
    idx_nm[torch.argsort(torch.abs(omega))[:6]] = False

    # Get eigenvalues (unit: eV/A^2/amu)
    eigenvals = omega[idx_nm]

    log.debug(
        "Eigenvalues (eV/A^2/amu): %.3f, %.3f %.3f, .... %.3f, %.3f %.3f"
        % (*eigenvals[:3], *eigenvals[-3:])
    )

    # Filter out unwanted modes (row equals normal mode, unit: eV/A^2/amu)
    V = V[:, idx_nm]

    # Get step
    step = _prfo_step(g_mass, V, eigenvals)

    # Multiply with inverse square mass to transform units to A
    step = (step * sqrtmmminv).reshape(-1, 3)

    return step


def activate_model_hessian(
    model: schnetpack.model.AtomisticModel,
) -> schnetpack.model.AtomisticModel:
    """
    Activate the computation of forces and Hessian in the model.

    Args:
        model (schnetpack.model.AtomisticModel): The model for which the Hessian matrix
            should be computed.

    Returns:
        schnetpack.model.AtomisticModel: The model with the Hessian computation
            activated.
    """

    # Define output for Hessians and forces.
    pred_forces_hessians = schnetpack.atomistic.Response(
        energy_key=properties.energy,
        response_properties=[properties.forces, properties.hessian],
    )
    hessian = False

    # Update output modules in a single pass
    new_output_modules = []
    for module in model.output_modules:
        if isinstance(
            module,
            (schnetpack.atomistic.response.Forces, schnetpack.atomistic.Response),
        ):
            if not hessian:
                # Replace with force and hessian output head
                new_output_modules.append(pred_forces_hessians)
                hessian = True
            # Skip adding duplicate response modules
        else:
            # Retain other modules as-is
            new_output_modules.append(module)

    # Raise an error if no Hessian computation module was added
    if not hessian:
        raise ValueError("Failed to activate Hessian computation")

    # Check for postprocessors that cast to double precision and remove them
    model.postprocessors = nn.ModuleList(
        [
            postprocessor
            for postprocessor in model.postprocessors
            if not isinstance(postprocessor, schnetpack.transform.CastTo64)
        ]
    )

    # Update the model's output modules
    model.output_modules = torch.nn.ModuleList(new_output_modules)

    # Update model derivatives and outputs
    model.collect_derivatives()
    model.collect_outputs()

    return model


class PhysicalTransform(trn.Transform):

    def __init__(
        self,
        calculator: Union[str, Calculator, schnetpack.model.AtomisticModel],
        output_key: str,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        if isinstance(calculator, (str, schnetpack.model.AtomisticModel)):
            self.calculator = self._load_model(calculator, device=device)
            self.model_flag = True
        elif isinstance(calculator, Calculator):
            self.calculator = calculator
            self.model_flag = False
        else:
            raise ValueError(
                "Calculator must be a (path to) Schnetpack model or ASE calculator."
            )

        self.output_key = output_key

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

        log.info("Activating force and Hessian computation...")
        model = activate_model_hessian(model)

        return model

    def _ase_hessian(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert isinstance(
            self.calculator, Calculator
        ), "Calculator must be an ase Calculator."
        # Convert to atoms an use ASE to get Hessian and forces
        atoms = inputs_to_atoms(inputs)
        atoms.calc = self.calculator
        hessian, forces, energy = _in_memory_hessian(atoms)

        energy = torch.from_numpy(energy)
        forces = torch.from_numpy(forces)
        hessian = torch.from_numpy(hessian)

        return hessian, forces, energy

    def _schnet_hessian(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert isinstance(
            self.calculator, nn.Module
        ), "Calculator must be a SchNetPack module."
        # Use SchNetPack to get Hessian and forces
        with torch.enable_grad():
            model_outputs = self.calculator(inputs)

        energy = model_outputs[properties.energy].detach()
        forces = model_outputs[properties.forces].detach()
        hessian = _reshape_hessian(
            forces.shape[0], model_outputs[properties.hessian]
        ).detach()

        return hessian, forces, energy

    def compute_hessian(
        self, inputs: Dict[str, torch.Tensor], epsilon: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input dtype and device
        dtype = inputs[properties.R].dtype
        device = inputs[properties.R].device

        if self.model_flag:
            hessian, forces, energy = self._schnet_hessian(inputs)
        else:
            hessian, forces, energy = self._ase_hessian(inputs)

        # Adapt device and dtype
        energy = energy.to(dtype=dtype, device=device)
        forces = forces.to(dtype=dtype, device=device)
        hessian = hessian.to(dtype=dtype, device=device)

        # Add regularization to Hessian to avoid singularities
        hessian += epsilon * torch.eye(hessian.shape[0], dtype=dtype, device=device)

        return hessian, forces, energy

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class PhysicalLoss(PhysicalTransform):
    """
    Generates physical velocities towards the transition state using Hessian and forces.
    """

    is_preprocessor: bool = False
    is_postprocessor: bool = True

    def __init__(
        self,
        calculator: Union[str, Calculator, schnetpack.model.AtomisticModel],
        output_key: str,
        prediction_key: str,
        device: str = "cpu",
    ) -> None:
        super().__init__(calculator, output_key, device=device)
        self.prediction_key = prediction_key

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}

        # Remove possible condition nodes for the calculator
        calculator_inputs = _remove_condition_nodes(inputs)

        # Overwrite positions with predicted positions
        calculator_inputs[properties.R] = inputs[self.prediction_key].clone().detach()

        # Get atomic masses and coordinates
        # coords = calculator_inputs[properties.R].clone().detach()
        # mass = torch.tensor(
        #     atomic_masses[calculator_inputs[properties.Z].cpu().numpy()],
        #     device=coords.device,
        #     dtype=coords.dtype,
        # )

        # Get hessian, forces, and energy
        hessian, forces, energy = self.compute_hessian(calculator_inputs)
        outputs[self.output_key] = energy

        # update the returned inputs.
        inputs.update(outputs)

        return inputs


class PhysicalVelocity(PhysicalTransform):
    """
    Generates physical velocities towards the transition state using Hessian and forces.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        calculator: Union[str, Calculator, schnetpack.model.AtomisticModel],
        output_key: str,
        time_key: str = "t",
        min_time: float = 0.0,
        cfm_velocity_key: str = "vel",
        device: str = "cpu",
        step_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ] = prfo_step_with_tr_projection,
        scale: bool = True,
    ) -> None:
        super().__init__(calculator, output_key, device=device)

        self.time_key = time_key
        self.min_time = min_time
        self.cfm_velocity_key = cfm_velocity_key
        self.step_fn = step_fn
        self.scale = scale

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}

        time_scale = 1.0
        # Check if time is greater than minimum time
        if self.time_key in inputs:
            t = inputs[self.time_key]
            if properties.conditions_mask in inputs:
                condition_mask = inputs[properties.conditions_mask]
                t = t[condition_mask == 1]
            t = torch.unique_consecutive(t)
            assert len(t) == 1, "Time must be unique for all atoms."
            time_scale = 1 - t
            if t < self.min_time:
                # Take CFM velocity
                outputs[self.output_key] = inputs[self.cfm_velocity_key].clone()

                # update the returned inputs.
                inputs.update(outputs)

                return inputs

        # Remove possible condition nodes for the calculator
        calculator_inputs = _remove_condition_nodes(inputs)

        # Get atomic masses and coordinates
        coords = calculator_inputs[properties.R].clone().detach()
        mass = torch.tensor(
            atomic_masses[calculator_inputs[properties.Z].cpu().numpy()],
            device=coords.device,
            dtype=coords.dtype,
        )

        # Get hessian, forces, and energy
        hessian, forces, _ = self.compute_hessian(calculator_inputs)

        # Get step using p-RFO or NR step function
        step = self.step_fn(hessian, forces, mass, coords)

        # Remove center of gravity
        step -= step.mean(dim=0)

        if self.scale:
            # Scale step to ensure useful integration
            step = step / step.norm() * inputs[self.cfm_velocity_key].norm() * 2

        if step.norm() > 10:
            log.warning(
                f"Huge step detected! Step: {step.norm():.3f}, max. force: {forces.max():.3f}, norm force {forces.norm():.3f}"
            )

        # Save detached step in outputs
        outputs[self.output_key] = step

        if properties.conditions_mask in inputs:
            condition_mask = inputs[properties.conditions_mask]

            # Add 0s for output (velocity)
            velocity = torch.zeros_like(inputs[properties.R])
            velocity[condition_mask == 1] = outputs[self.output_key]
            outputs[self.output_key] = velocity

        # update the returned inputs.
        inputs.update(outputs)

        return inputs


class NormalModeSampling(trn.Transform):
    """
    Generate random displacements based on normal modes.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self, weight_type: str = "temperature", temperature: float = 300.0):
        super().__init__()
        self.weight_type = weight_type
        self.temperature = temperature

    def get_mode_weights(
        self, energy: torch.Tensor, temperature: float, weight_type: str = "temperature"
    ):
        if weight_type == "uniform":
            ci = torch.ones(len(energy), device=energy.device, dtype=energy.dtype)
        elif weight_type == "boltzmann":
            ci = torch.exp(-energy / temperature)
        elif weight_type == "temperature":
            ci = temperature / (torch.abs(energy) + 1e-6)
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
        return ci

    def get_displacement(
        self,
        modes: torch.Tensor,
        energy: torch.Tensor,
        temperature: float = 300.0,
        weight_type: str = "temperature",
    ):
        """
        Generate random displacements based on normal modes. Displacement is calculated
        by setting a hamornic potential to a randomly scaled thermal energy at
        temperature T and solving it for the displacement Ri:
        
        ci * k * T = 1/2 * K*i * Ri^2 —> solving for Ri lead to:
        Ri = sqrt( 2* ci * k * T / Ki)
        """
        # Extract degrees of freedom
        Nf = modes.shape[0]

        # Get random numbers to weight each normal mode (normalize that sum_i ci in [0,1])
        Nf = len(energy)
        ci = torch.rand(Nf, device=modes.device, dtype=modes.dtype)
        ci /= ci.sum() + torch.rand(1, device=modes.device, dtype=modes.dtype)

        # Get random signs that follow a Bernoulli distribution with P=0.5
        signs = 2 * torch.bernoulli(torch.full_like(ci, 0.5)) - 1

        # Scaling factor based on temperature and energy of the mode
        mode_weights = self.get_mode_weights(
            energy, temperature=units.kB * temperature, weight_type=weight_type
        )
        Ri = torch.sqrt(2 * ci * mode_weights) * signs

        # Get the displacement (sum over weighted modes)
        displacement = torch.sum(modes * Ri.unsqueeze(1).unsqueeze(1), dim=0)

        return displacement

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        modes = inputs[properties.eig_modes]
        energies = inputs[properties.eig_energies]
        displacement = self.get_displacement(
            modes, energies, self.temperature, self.weight_type
        )

        # Remove center of gravity 
        displacement = displacement - displacement.mean(dim=-2)
        inputs[properties.nm_distplacement] = displacement

        return inputs
