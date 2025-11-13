import os
import logging
from typing import List, Optional

from ase import Atoms
from ase.io import write

from tqdm import tqdm
from aefm.sampling import Sampler
from aefm.utils import batch_inputs_to_atoms, get_rmsd

import numpy as np
import schnetpack.transform as trn
import ase
from aefm.transform import (
    AllToAllNeighborList,
    SubtractCenterOfGeometry,
)
from schnetpack.interfaces.ase_interface import AtomsConverter
import time

log = logging.getLogger(__name__)

class AEFMSampler:
    def __init__(self, 
                 sampler: Sampler,
                 store_path: str, 
                 converter: Optional[AtomsConverter]=None, 
                 reference_path: Optional[str]=None, 
                 save_trajectory: bool=True, 
                 identifier: List[str]=None
        ):
        """
            Initializes the sampler with the given model, sampler, storage path, and
            device.
            Args:
                sampler: Sampler instance used for sampling.
                store_path: Path where samples or results will be stored.
                converter: Optional AtomsConverter for converting ASE Atoms objects to
                    model inputs. If None, a default converter will be created.
                reference: Optional path to a reference ASE db for comparison. Needs 
                    "rxn" as key to match the samples.
                reference_path: Optional path to a reference ASE Atoms object for
                    RMSD calculation.
                save_trajectory: Whether to save the trajectory of the samples.
        """
        self.sampler = sampler
        self.store_path = store_path
        self.reference = {atoms.info["rxn"]: atoms for atoms in ase.io.read(reference_path, index=":")} if reference_path is not None else None
        self.identifier = identifier if identifier is not None else ["rxn"]
        self.reference = None
        if reference_path is not None:
            if not os.path.exists(reference_path):
                raise ValueError(f"Reference path <{reference_path}> does not exist.")
            log.info(f"Reading reference database from <{reference_path}>...")
            self.reference = {}
            for atoms in ase.io.read(reference_path, index=":"):
                atom_identifier = self.get_atom_identifier(atoms)
                self.reference[atom_identifier] = atoms
        
        if converter is not None:
            self.converter = converter
        else:
            # Create a default converter
            self.converter = AtomsConverter(
                neighbor_list=AllToAllNeighborList(),
                transforms=[
                    trn.CastTo64(),
                    SubtractCenterOfGeometry(),
                    trn.CastTo32(),
                ],
                device=sampler.device,
            )
        self.converter.device = sampler.device
        self.save_trajectory = save_trajectory
        log.info(f"Using device: {self.sampler.device}")
        log.info(f"Storing samples at: <{self.store_path}> using identifier: {self.identifier}")
        log.info(
            f"Using {self.sampler.fixpoint_algorithm} with following settings:\n"+
            "\n".join(f"   {k}: {v}" for k, v in self.sampler.fixpoint_settings.items())
        )

    def get_atom_identifier(self, sample: Atoms, rxn: int=None) -> str:
        """Get a unique identifier for the sample based on the specified keys."""
        return "_".join([key + "_" + str(sample.info[key]) for key in self.identifier])
    
    def save_samples(
        self,
        samples: List[Atoms],
        path_keywords: List[str]=[],
    ):
        """
        Saves the samples to disk.

        Args:
            samples: List of ASE Atoms of the optimization trajectory to be saved.
            aligned_sample: Optional ASE Atoms of the aligned sample to be saved.
            path_keywords: List[str]: List of keywords to be used for the path.
        """

        # Define store path
        store_path = os.path.join(
            self.store_path, *path_keywords
        )

        for sample in samples:
            # Create for each sample an own folder
            atom_identifier = sample[-1].info["atom_identifier"]
                
            # Save trajectory
            sample_path = os.path.join(store_path, atom_identifier)
            os.makedirs(sample_path, exist_ok=True)
            write(os.path.join(sample_path, "trajectory.xyz"), sample)

    def sample(self, samples: List[ase.Atoms]):
        
        rmsd_initial = []
        rmsd_final = []
        n_iterations = []
        times = []
        
        db_path = os.path.join(self.store_path, "samples.xyz")
        if os.path.exists(db_path):
            logging.warning(
                f"Samples already exist at {self.store_path}. "
                "This will overwrite the existing samples."
            )
            os.remove(db_path)
        
        with tqdm(samples, desc="Processing", unit="sample") as pbar:
            for i, sample in enumerate(pbar):
                atom_identifier = self.get_atom_identifier(sample, i)
                if self.reference and atom_identifier in self.reference:
                    ref_atoms = self.reference[atom_identifier]
                    if len(ref_atoms) != len(sample):
                        raise ValueError(f"Reference and sample have different number of atoms for reaction {atom_identifier}.")
                    _, rmsd_init = get_rmsd(sample=sample, reference=ref_atoms, same_order=True)
                    rmsd_initial.append(rmsd_init)
                    
                batch = self.converter(sample)
                start_time = time.time()
                nfe, results = self.sampler(batch)
                end_time = time.time()
                sampling_time = end_time - start_time
                times.append(sampling_time)
                n_iterations.append(nfe)
                
                # This is a list of iterations for all molecules at each iteration
                trajectory = [batch_inputs_to_atoms(result) for result in results]
                
                # This is a list of molecules at each iteration
                trajectory_atoms = []
                for i in range(len(trajectory[0])):
                    trajectory_atoms.append([traj[i] for traj in trajectory])
                
                # Check the last atom in the trajectory
                final_sample = trajectory_atoms[0][-1]
                final_sample.info.update(sample.info)
                final_sample.info.pop("rmsd", None) # remove previous rmsd if exists
                final_sample.info.pop("delta_e", None) # remove previous delta_e if exists
                final_sample.info["sampler"] = "AEFM"
                final_sample.info["n_steps"] = nfe
                final_sample.info["sampling_time"] = round(sampling_time,2)
                final_sample.info["atom_identifier"] = atom_identifier
                metrics = {"n_steps": f"{nfe:.0f}", "time": f"{sampling_time:.2f}"}
                if self.reference and atom_identifier in self.reference:
                    try:
                        final_sample, rmsd = get_rmsd(sample=final_sample, reference=ref_atoms, same_order=True)
                    except:
                        rmsd = float("nan")
                        log.warning(f"Could not align sample to reference for reaction {atom_identifier}.")
                    rmsd_final.append(rmsd)
                    metrics["rmsd_final"] = f"{rmsd:.3f}"
                    metrics["rmsd_initial"] = f"{rmsd_init:.3f}"
                    metrics["rmsd_improvement"] = f"{(rmsd_init - rmsd)/rmsd_init*100:.2f}"
                
                pbar.set_postfix(metrics, refresh=False)

                if self.save_trajectory:
                    # Save the trajectory for each sample
                    self.save_samples(trajectory_atoms)

                # Save the final (aligned) sample in database
                write(
                    db_path,
                    final_sample,
                    append=True,
                )
        
        stats_dict = {
            "n_iterations": n_iterations,
            "times": times,
        }
        
        log.info(
            "Number of iterations per sample:\n " +
            f"Mean: {np.mean(n_iterations):.0f} \n " +
            f"Median: {np.median(n_iterations):.0f}"
        )
        
        log.info(
            "Time per sample (s):\n "+
            f"Mean: {np.mean(times):.2f} \n " +
            f"Median: {np.median(times):.2f}"
        )
        
        if len(rmsd_initial) > 0:
            stats_dict["rmsd_initial"] = rmsd_initial
            stats_dict["rmsd_final"] = rmsd_final
            log.info(
                f"Initial RMSD:\n Mean: {np.mean(rmsd_initial):.3f} \n " +
                f"Median: {np.median(rmsd_initial):.3f}"
            )
            log.info(
                f"Final RMSD:\n Mean: {np.mean(rmsd_final):.3f} \n " +
                f"Median: {np.median(rmsd_final):.3f}"
            )
            
        np.savez( 
            os.path.join(self.store_path, "stats.npz"),
            **stats_dict
        )

            
