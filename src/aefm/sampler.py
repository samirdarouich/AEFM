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

log = logging.getLogger(__name__)

class AEFMSampler:
    def __init__(self, sampler: Sampler, store_path: str, converter: Optional[AtomsConverter]=None, reference_path: Optional[str]=None, save_trajectory: bool=True):
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
        self.offset = 0
        self.save_trajectory = save_trajectory
        log.info(f"Using device: {self.sampler.device}")
        log.info(
            f"Using {self.sampler.fixpoint_algorithm} with following settings:\n"+
            "\n".join(f"   {k}: {v}" for k, v in self.sampler.fixpoint_settings.items())
        )

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
            rxn = sample[0].info.get("rxn", self.offset)
            sample_path = os.path.join(store_path, f"reaction_{rxn}")
            os.makedirs(sample_path, exist_ok=True)

            # Save trajectory
            write(os.path.join(sample_path, "trajectory.xyz"), sample)
            
            self.offset += 1

    def sample(self, samples: List[ase.Atoms]):
        
        rmsd_initial = []
        rmsd_final = []
        n_iterations = []
        
        db_path = os.path.join(self.store_path, "samples.xyz")
        if os.path.exists(db_path):
            logging.warning(
                f"Samples already exist at {self.store_path}. "
                "This will overwrite the existing samples."
            )
            os.remove(db_path)
        
        with tqdm(samples, desc="Processing", unit="sample") as pbar:
            for sample in pbar:
                rxn = sample.info.get("rxn", None)
                if self.reference and rxn in self.reference:
                    ref_atoms = self.reference[rxn]
                    if len(ref_atoms) != len(sample):
                        raise ValueError(f"Reference and sample have different number of atoms for reaction {rxn}.")
                    _, rmsd_init = get_rmsd(sample=sample, reference=ref_atoms, same_order=True)
                    rmsd_initial.append(rmsd_init)
                    
                batch = self.converter(sample)
                nfe, results = self.sampler(batch)
                n_iterations.append(nfe)
                
                # This is a list of iterations for all molecules at each iteration
                trajectory = [batch_inputs_to_atoms(result) for result in results]
                
                # This is a list of molecules at each iteration
                trajectory_atoms = []
                for i in range(len(trajectory[0])):
                    trajectory_atoms.append([traj[i] for traj in trajectory])
                
                # Check the last atom in the trajectory
                final_sample = trajectory_atoms[0][-1]
                final_sample.info["n_steps"] = nfe
                metrics = {"n_steps": f"{nfe:.0f}"}
                if self.reference and rxn in self.reference:
                    try:
                        final_sample, rmsd = get_rmsd(sample=final_sample, reference=ref_atoms, same_order=True)
                    except:
                        rmsd = float("nan")
                        log.warning(f"Could not align sample to reference for reaction {rxn}.")
                    rmsd_final.append(rmsd)
                    metrics["rmsd_final"] = f"{rmsd:.3f}"
                    metrics["rmsd_initial"] = f"{rmsd_init:.3f}"
                    trajectory_atoms[0][0].info["rxn"] = rxn
                    final_sample.info["rxn"] = rxn
                
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
            
        if len(rmsd_initial) > 0:
            log.info(
                "Number of iterations per sample:\n " +
                f"Mean: {np.mean(n_iterations):.0f} \n " +
                f"Median: {np.median(n_iterations):.0f}"
            )
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
                rmsd_initial=rmsd_initial,
                rmsd_final=rmsd_final,
                n_iterations=n_iterations
            )

            
