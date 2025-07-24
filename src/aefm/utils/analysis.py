import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import py3Dmol
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.units import Bohr, Hartree
from pymatgen.analysis.molecule_matcher import (
    BruteForceOrderMatcher,
    GeneticOrderMatcher,
    HungarianOrderMatcher,
    KabschMatcher,
)
from pymatgen.core import Molecule
from pyscf import dft, gto

from aefm import properties
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo
import os
from functools import partial
log = logging.getLogger(__name__)


def _quaternion_to_matrix(q):
    """Returns a rotation matrix computed from a unit quaternion.
    Input: (4,) torch tensor.
    """
    q0, q1, q2, q3 = q
    R_q = torch.tensor(
        [
            [
                q0**2 + q1**2 - q2**2 - q3**2,
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                q0**2 - q1**2 + q2**2 - q3**2,
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                q0**2 - q1**2 - q2**2 + q3**2,
            ],
        ],
        device=q.device,
        dtype=q.dtype,
    )
    return R_q


def _rotation_matrix_from_points(m0, m1):
    """Returns a rigid transformation/rotation matrix that minimizes the
    RMSD between two set of points.

    m0 and m1 should be (3, npoints) torch tensors with
    coordinates as columns.

    The centroids should be set to the origin prior to
    computing the rotation matrix.
    """
    v0 = m0.clone()
    v1 = m1.clone()

    # Compute the rotation quaternion
    R11, R22, R33 = torch.sum(v0 * v1, dim=1)
    R12, R23, R31 = torch.sum(v0 * torch.roll(v1, shifts=-1, dims=0), dim=1)
    R13, R21, R32 = torch.sum(v0 * torch.roll(v1, shifts=-2, dims=0), dim=1)

    F = torch.tensor(
        [
            [R11 + R22 + R33, R23 - R32, R31 - R13, R12 - R21],
            [R23 - R32, R11 - R22 - R33, R12 + R21, R13 + R31],
            [R31 - R13, R12 + R21, -R11 + R22 - R33, R23 + R32],
            [R12 - R21, R13 + R31, R23 + R32, -R11 - R22 + R33],
        ],
        device=v0.device,
        dtype=v0.dtype,
    )

    # Compute eigenvalues and eigenvectors
    w, V = torch.linalg.eigh(F)
    q = V[:, torch.argmax(w)]

    # Compute the rotation matrix from quaternion
    R = _quaternion_to_matrix(q)

    return R


def _ase_align(p: torch.Tensor, p0: torch.Tensor):
    """align p to p0 using the Kabsch algorithm."""

    # centeroids to origin
    c = torch.mean(p, dim=0)
    p -= c
    c0 = torch.mean(p0, dim=0)
    p0 -= c0

    R = _rotation_matrix_from_points(p.T, p0.T)

    return torch.mm(p, R.T) + c0


def get_total_brute_force_permutations(
    atomic_numbers: Union[List[int], Tuple[int], np.ndarray],
) -> int:
    """
    Calculate the total number of permutations based on the occurrence of each unique atom.

    Args::
        atomic_numbers (Union[List[int], np.ndarray]):
          List of atomic numbers.

    Returns:
        int:
          Total number of permutations.
    """
    # Get occurence of each unique atom
    _, count = np.unique(atomic_numbers, return_counts=True)
    total_permutations = 1
    for c in count:
        total_permutations *= math.factorial(c)
    return total_permutations


def pymatgen_align(
    sample: Atoms,
    target: Atoms,
    same_order: bool = True,
    max_permutations: int = 1e4,
) -> Atoms:
    """
    Aligns the sample molecule to the reference molecule using various matching algorithms.

    In case the same atom order is given, simply use the Kabsch Matcher algorithm.
    Otherwise 3 alternative algorithms are used:
      1) Brute force order matching if the total number of possible permuations of atoms
         is lower than 1e5. This test every permutation of atoms with the same species.
         For each trial, the Kabsch Matcher algorithm is used, and the permutation with
         the lowest rmsd is taken

    Args:
        sample: The sample which will be translated/rotated to match the target with
            lowest RMSD.
        reference: Target to match the sample with lowest RMSD.
        same_order: Whether the atoms in both molecules are in the same order. In case
            they do not have the same order, a brute force search is performed to find
            the best permutation (only permutes alike atom types).
        max_permutations: Maximum number of permutations to test if the order of atoms
            is different.

    Returns:
        Atoms:
          The aligned sample molecule.
    """
    aligned_sample_ase = sample.copy()

    sample_pymatgen = Molecule(
        species=sample.get_atomic_numbers(), coords=sample.get_positions()
    )
    target_pymatgen = Molecule(
        species=target.get_atomic_numbers(), coords=target.get_positions()
    )

    if same_order:
        assert np.all(
            sample.get_atomic_numbers() == target.get_atomic_numbers()
        ), "Expected sample and target to have the same atom ordering."
        log.debug("Use Kabsch Matcher matching.")
        bfm = KabschMatcher(target_pymatgen)
        aligned_sample, _ = bfm.fit(sample_pymatgen)
    else:
        total_permutations = get_total_brute_force_permutations(
            sample_pymatgen.atomic_numbers
        )

        if total_permutations < max_permutations:
            log.debug("Use brute force matching.")
            bfm = BruteForceOrderMatcher(target_pymatgen)
            aligned_sample, _ = bfm.fit(sample_pymatgen)
        else:
            bfm = GeneticOrderMatcher(target_pymatgen, threshold=0.5)
            pairs = bfm.fit(sample_pymatgen)
            if len(pairs) == 0:
                log.debug("Use hungarian order matching.")
                bfm = HungarianOrderMatcher(target_pymatgen)
                aligned_sample, _ = bfm.fit(sample_pymatgen)
            else:
                log.debug("Use genetic order matching.")
                min_idx = np.argmin([p[1] for p in pairs])
                aligned_sample = [p[0] for p in pairs][min_idx]

    aligned_sample_ase.set_positions(aligned_sample.cart_coords)
    return aligned_sample_ase


def _compute_rmsd(atom1: Atoms, atom2: Atoms) -> np.ndarray:
    """
    Compute the RMSD between two atomic structures.
    (https://en.wikipedia.org/wiki/Root_mean_square_deviation_of_atomic_positions)

    RMSD(v,w) = sqrt( 1/n sum_i^n ||v_i-w_i||^2 )
              = sqrt( 1/n sum_i^n ( (v_i,x-w_i,x)^2 + (v_i,y-w_i,y)^2 + (v_i,z-w_i,z)^2 )

    """
    if sorted(atom1.get_atomic_numbers()) != sorted(atom2.get_atomic_numbers()):
        raise ValueError("The number of the same species aren't matching!")
    return ((atom1.positions - atom2.positions) ** 2).sum(-1).mean() ** 0.5


def get_rmsd(
    sample: Atoms,
    reference: Atoms,
    same_order: bool = True,
    ignore_chirality: bool = False,
    max_permutations: int = 1e6,
) -> Tuple[Atoms, np.ndarray]:
    """Calculates the RMSD between two molecular structures with optional chirality reflection.

    Args:
        sample: The sample which will be translated/rotated to match the target with
            lowest RMSD.
        target: Target to match the sample with lowest RMSD.
        same_order: Whether the atoms in both molecules are in the same order. In case
            they do not have the same order, a brute force search is performed to find
            the best permutation (only permutes alike atom types).
        ignore_chirality: If True, the function also calculates RMSD for the molecule
            reflected along the z-axis, and returns the minimum structure and RMSD.
        max_permutations: Maximum number of permutations to test if the order of atoms
            is different.

    Returns:
        Tuple[Atoms, float]:
          The aligned molecule and minimum RMSD, considering chirality if specified.
    """
    # align and compute the RMSD
    aligned_sample = pymatgen_align(sample, reference, same_order, max_permutations)
    rmsd = _compute_rmsd(reference, aligned_sample)
    log.debug(f"RMSD: {rmsd:.4f}")

    if ignore_chirality:
        # Reflect coordinates in z-axis, align and compute RMSD
        sample_reflected = sample.copy()
        reflected_pos = sample_reflected.get_positions()
        reflected_pos[:, -1] = -reflected_pos[:, -1]
        sample_reflected.set_positions(reflected_pos)

        # Compure RMSD for refrecleted molecule
        aligned_sample_reflect = pymatgen_align(
            sample_reflected, reference, same_order, max_permutations
        )
        rmsd_reflect = _compute_rmsd(reference, aligned_sample_reflect)
        log.debug(f"RMSD reflected: {rmsd_reflect:.4f}")

        # Return lower RMSD
        if rmsd_reflect < rmsd:
            rmsd = rmsd_reflect
            aligned_sample = aligned_sample_reflect

    # Add RMSD to the info of the aligned molecule
    aligned_sample.info["rmsd"] = rmsd

    return aligned_sample, rmsd


def inputs_to_atoms(inputs: Dict[str, torch.Tensor]) -> Atoms:
    """
    Converts a SchNetPack input to an ASE Atoms object.

    Args:
        inputs: The input in SchNetPack format.

    Returns:
        Atoms: The ASE Atoms object.
    """
    R = inputs[properties.R].detach().cpu().numpy()
    Z = inputs[properties.Z].detach().cpu().numpy()

    if properties.conditions_mask in inputs:
        t_mask = (
            inputs[properties.conditions_mask].detach().cpu().nonzero()[:, 0].numpy()
        )
        R = R[t_mask]
        Z = Z[t_mask]

    if properties.reaction_ids in inputs and properties.image_type in inputs:
        info = {
            "reaction_id": inputs[properties.reaction_ids].item(),
            "origin_image_type": properties.IMAGE_TYPES_REVERSE[
                int(inputs[properties.image_type].item())
            ],
        }
    else:
        info = None

    atoms = Atoms(positions=R, numbers=Z, info=info)

    if properties.energy in inputs:
        if properties.forces in inputs:
            calc = SinglePointCalculator(
                atoms,
                energy=float(inputs[properties.energy].detach().cpu().item()),
                forces=inputs[properties.forces].detach().cpu().numpy(),
            )
        else:
            calc = SinglePointCalculator(
                atoms, energy=float(inputs[properties.energy].detach().cpu().item())
            )

        atoms.calc = calc
    return atoms


def batch_inputs_to_atoms(inputs: Dict[str, torch.Tensor]) -> List[Atoms]:
    """
    Converts a batch of inputs to a list of ASE Atoms objects.

    Args:
        inputs: The input batch in SchNetPack format.

    Returns:
        List[Atoms]: The list of ASE Atoms objects.
    """
    atoms_list = []

    if properties.conditions_mask in inputs:
        target_mask = inputs[properties.conditions_mask]

    for m in inputs[properties.idx_m].unique():
        mask = inputs[properties.idx_m] == m
        R = inputs[properties.R][mask].detach().cpu().numpy()
        Z = inputs[properties.Z][mask].detach().cpu().numpy()

        if properties.conditions_mask in inputs:
            t_mask = target_mask[mask].nonzero()[:, 0].detach().cpu().numpy()
            R = R[t_mask]
            Z = Z[t_mask]

        if properties.reaction_ids in inputs and properties.image_type in inputs:
            info = {
                "reaction_id": inputs[properties.reaction_ids][m].item(),
                "origin_image_type": properties.IMAGE_TYPES_REVERSE[
                    int(inputs[properties.image_type][m].item())
                ],
            }
        else:
            info = None

        atoms = Atoms(positions=R, numbers=Z, info=info)

        if properties.energy in inputs:
            if properties.forces in inputs:
                forces = inputs[properties.forces][mask].detach().cpu().numpy()
                if properties.conditions_mask in inputs:
                    forces = inputs[properties.forces][t_mask].detach().cpu().numpy()

                calc = SinglePointCalculator(
                    atoms,
                    energy=float(inputs[properties.energy][m].detach().cpu().item()),
                    forces=forces,
                )
            else:
                calc = SinglePointCalculator(
                    atoms,
                    energy=float(inputs[properties.energy][m].detach().cpu().item()),
                )

            atoms.calc = calc

        atoms_list.append(atoms)
    return atoms_list


def batch_rmsd(
    references: torch.Tensor,
    samples: Dict[str, torch.Tensor],
    same_order: bool = False,
    ignore_chirality: bool = False,
    max_permutations: int = 1e6,
) -> Tuple[List[Atoms], List[Atoms], torch.Tensor]:
    """
    Computes the RMSD between a batch of reference and sampled/denoised positions.

    Args:
        references: The reference positions.
        samples: Sampled or denoised positions.
        same_order: Whether the atoms in both molecules are in the same order. In case
            they do not have the same order, a brute force search is performed to find
            the best permutation (only permutes alike atom types).
        ignore_chirality: If True, the function also calculates RMSD for the molecule
            reflected along the z-axis, and returns the minimum structure and RMSD.
        max_permutations: Maximum number of permutations to test if the order of atoms
            is different.

    Returns:
        Tuple[List[Atoms], List[Atoms], torch.Tensor]:
          The aligned molecules, the original molecules, and the RMSDs.
    """
    rmsds = []
    aligned_mols = []
    mols = []

    if properties.conditions_mask in samples:
        target_mask = samples[properties.conditions_mask]

    # loop over molecules/systems
    for m in samples[properties.idx_m].unique():
        # get the indices of the current molecule
        mask = samples[properties.idx_m] == m

        # get the positions and atomic numbers
        R_0 = references[mask].detach().cpu().numpy()
        R = samples[properties.R][mask].detach().cpu().numpy()
        Z = samples[properties.Z][mask].detach().cpu().numpy()

        # Remove conditioning nodes from samples
        if properties.conditions_mask in samples:
            t_mask = target_mask[mask].nonzero()[:, 0].detach().cpu().numpy()
            R = R[t_mask]
            Z = Z[t_mask]
            R_0 = R_0[t_mask]

        if properties.reaction_ids in samples and properties.image_type in samples:
            info = {
                "reaction_id": samples[properties.reaction_ids][m].item(),
                "origin_image_type": properties.IMAGE_TYPES_REVERSE[
                    int(samples[properties.image_type][m].item())
                ],
            }
        else:
            info = None

        # create ase.Atoms objects
        ref_mol = Atoms(positions=R_0, numbers=Z)
        mol = Atoms(positions=R, numbers=Z, info=info)

        # compute the rmsd or set it to NaN if fails, e.g. for very different structures
        try:
            aligned_mol, rmsd = get_rmsd(
                mol, ref_mol, same_order, ignore_chirality, max_permutations
            )
        except Exception:
            aligned_mol, rmsd = mol, torch.inf

        rmsds.append(rmsd)
        aligned_mols.append(aligned_mol)
        mols.append(mol)

    return aligned_mols, mols, torch.tensor(rmsds, device=samples[properties.R].device)


def atoms_to_xyz_text(atoms: Atoms):
    xyz_str = f"{len(atoms)}\n\n"
    for atom, pos in zip(atoms, atoms.positions):
        xyz_str += f"{atom.symbol} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n"  # type: ignore
    return xyz_str


def visualize_atoms_list(
    atoms_list: Sequence[Union[Atoms, str]],
    colors: Optional[List[str]] = None,
    style_dicts: Optional[List[Dict[str, Any]]] = None,
) -> py3Dmol.view:
    """
    Visualizes a list of atomic structures represented by Atoms objects using py3Dmol.

    Args:
        atoms_list: List of atoms objects | xyz paths representing atomic structures to
            visualize.
        colors: List of colors to assign to each structure.
        style_dicts: List of style dictionaries to assign to each structure.

    Returns:
      py3Dmol.view:
        The html view object of py3Dmol.
    """
    xyzs = []
    for atoms in atoms_list:
        if isinstance(atoms, Atoms):
            xyzs.append(atoms_to_xyz_text(atoms))
        elif isinstance(atoms, str):
            if ".xyz" not in atoms:
                raise ValueError("Expected xyz file!.")
            with open(atoms) as f:
                xyzs.append(f.read())
        else:
            raise ValueError("Either specify atoms object or xyz file")

    view = py3Dmol.view(width=800, height=400)

    default_style = {"stick": {}, "sphere": {"radius": 0.36}}
    for i, xyz in enumerate(xyzs):
        view.addModel(xyz, "xyz")
        if style_dicts is not None:
            style_dict = style_dicts[i]
        else:
            style_dict = default_style

        if colors is not None:
            if "stick" not in style_dict:
                style_dict["stick"] = {"color": colors[i]}
            else:
                style_dict["stick"].update({"color": colors[i]})

        view.setStyle(
            {"model": i},
            style_dict,
        )
    view.zoomTo()
    return view


def visualize_reaction(atoms_list: Sequence[Union[Atoms, str]], offset: float = 5.0):
    """
    Visualize a chemical reaction given a list of ASE `Atoms` or paths to xyz files.

    This function takes a list of atomic structures (from the ASE `Atoms` class)
    and shifts each structure along one axis by a specified offset, relative to
    its index in the list. The function then visualizes the shifted atomic structures.

    Args:
        atoms_list: A list of atomic structures to visualize. Each element is an ASE
            `Atoms` object or path to a structure file that ASE can read.
        offset: The distance to shift each atomic structure. The i-th structure in the
            list is shifted by `i * offset`.

    Returns:
        py3Dmol.view:
          The html view object of py3Dmol.
    """
    shifted_atoms = []
    for i, atoms in enumerate(atoms_list):
        if isinstance(atoms, Atoms):
            shifted_atom = atoms.copy()
        elif isinstance(atoms, str):
            shifted_atom = read(atoms)
        shifted_atom.translate(offset * i)  # type: ignore
        shifted_atoms.append(shifted_atom)
    return visualize_atoms_list(shifted_atoms)


def atoms_to_pyscf(atoms: Atoms):
    xyz_str = ""
    for atom, pos in zip(atoms, atoms.positions):
        xyz_str += f"{atom.symbol} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}; "
    return xyz_str

def get_pyscf_kernel(atom: Atoms | str,
    functional: str,
    basis: str,
    settings: Optional[Dict[str, Any]] = None,):
    
    if settings is None:
        settings = {
            "conv_tol": 1e-6,
            "max_cycle": 200,
            "max_memory": 32000,
        }

    if isinstance(atom, Atoms):
        mol = gto.M(
            atom=atoms_to_pyscf(atom),
            unit="Ang",
            basis=basis,
        )
    elif isinstance(atom, str):
        atom = read(atom)
        mol = gto.M(
            atom=atoms_to_pyscf(atom),
            unit="Ang",
            basis=basis,
        )

    mol.build()

    mf = dft.RKS(mol)
    mf.xc = functional

    for key, value in settings.items():
        setattr(mf, key, value)
    
    return mf
    
def get_sp_energy(
    atom: Atoms | str,
    functional: str,
    basis: str,
    forces: bool = False,
    settings: Optional[Dict[str, Any]] = None,
) -> Atoms:

    mf = get_pyscf_kernel(atom, functional, basis, settings)
    mf.run()

    # Save results
    atom = atom.copy()
    results = {}

    # Convert from Hartree to eV
    e_tot = mf.e_tot * Hartree
    results["energy"] = e_tot

    if mf.converged and forces:
        # Convert from Hartree/Bohr to eV/Angstrom
        force = -1.0 * mf.nuc_grad_method().kernel() * Hartree / Bohr
        results["forces"] = force

    atom.calc = SinglePointCalculator(
        atoms=atom,
        **results,
    )

    return atom

def save_trajectory(envs, atom, output_folder):
    mol = envs.get('mol')
    if mol is not None:
        atom = atom.copy()
        atom.positions = mol.atom_coords() * Bohr
        write(f"{output_folder}/opt_trajectory.xyz", atom, append=True)
            
def geom_opt(atom: Atoms | str, functional: str, basis: str, transition: bool=True, forces: bool=False, hessian: bool=True, settings: Optional[Dict[str, Any]] = None, conv_params: Optional[Dict[str, Any]] = None, output_folder:Optional[str]=None):
    
    if isinstance(atom, str):
        atom = read(atom)
        
    if output_folder is None:
        output_folder = "."
    else:
        os.makedirs(output_folder, exist_ok=True)
        
    # Define optimization settings
    if conv_params is None:
        conv_params = {"maxsteps": 300}
    
    # Optimize and save the optimized structure
    mf = get_pyscf_kernel(atom,functional=functional, basis=basis, settings=settings)
    mol_opt = optimize(mf, maxsteps=300, transition=transition, callback=partial(save_trajectory, atom=atom,output_folder=output_folder), **conv_params)
    mol_opt.tofile(f"{output_folder}/opt_sample.xyz", format="xyz")
    
    # Read the optimized structure and get SP energy (& forces)
    mf = get_pyscf_kernel(f"{output_folder}/opt_sample.xyz",functional=functional, basis=basis, settings=settings)
    mf.run()
    
    # Save optimized mol as ase atoms object
    atom = atom.copy()
    atom.positions = mf.mol.atom_coords() * Bohr
    results = {}

    # Convert from Hartree to eV
    e_tot = mf.e_tot * Hartree
    results["energy"] = e_tot
    
    if mf.converged and forces:
        # Convert from Hartree/Bohr to eV/Angstrom
        force = -1.0 * mf.nuc_grad_method().kernel() * Hartree / Bohr
        results["forces"] = force
    
    if hessian:
        # Get Hessian
        hess = mf.Hessian().kernel()
        freq_info = thermo.harmonic_analysis(mf.mol, hess)
    
        # Extract force constants and normal modes 
        # Convert from Hartree/Bohr^2 to eV/Angstrom^2
        force_const = freq_info["force_const_au"] * Hartree / Bohr**2
        normal_modes = freq_info["norm_mode"]
        np.savez(f"{output_folder}/hessian_stats.npz",force_const=force_const, normal_modes=normal_modes)
        atom.arrays["force_const"] = force_const
    
    atom.calc = SinglePointCalculator(
        atoms=atom,
        **results,
    )
    
    return atom
    