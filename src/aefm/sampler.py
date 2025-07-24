import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from ase import Atoms
from ase.io import write
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from torch import nn

from aefm import properties
from aefm.sampling import Sampler
from aefm.utils import batch_inputs_to_atoms, get_rmsd
import os

import numpy as np
import schnetpack
import schnetpack.transform as trn
import torch
import torch.nn as nn
from ase.io import read, write
import ase
from aefm.transform import (
    AllToAllNeighborList,
    SubtractCenterOfGeometry,
)
from schnetpack.interfaces.ase_interface import AtomsConverter


class AEFMSampler:
    def __init__(self, sampler: Sampler, store_path: str, reference_path: Optional[str]=None):
        """
            Initializes the sampler with the given model, sampler, storage path, and device.
            Args:
                sampler: Sampler instance used for sampling.
                store_path: Path where samples or results will be stored.
                reference: Optional path to a reference ASE db for comparison. Needs "rxn" as key to match the samples.
        """
        self.sampler = sampler
        self.store_path = store_path
        self.reference = {atoms.info["rxn"]: atoms for atoms in ase.io.read(reference_path, index=":")} if reference_path is not None else None
        self.converter = AtomsConverter(
            neighbor_list=AllToAllNeighborList(),
            transforms=[
                trn.CastTo64(),
                SubtractCenterOfGeometry(),
                trn.CastTo32(),
            ],
            device=sampler.device,
        )
    
    def save_samples(
        self,
        samples: List[Atoms],
        keyword: str,
        path_keywords: List[str],
        test=False,
        search_for_offset=True,
    ):
        """
        Saves the samples to disk.

        Args:
            samples: List of ASE Atoms of the final samples to be saved.
            keyword: keyword to be used for the file name.
            path_keywords: List of keywords to be used for the path.
            search_for_offset: whether to search for an offset in the sample folder
                names. If not it will take the class attribute offset.
        """

        # Define store path
        store_path = os.path.join(
            self.store_path, *path_keywords
        )

        # Define offset
        if search_for_offset:
            self.offset = 0
            if os.path.exists(store_path):
                files = os.listdir(store_path)
                if len(files) > 0:
                    self.offset = (
                        max(
                            [
                                int(file.split("_")[-1])
                                for file in files
                                if "sample_" in file
                            ]
                        )
                        + 1
                    )

        for idx, sample in enumerate(samples):
            # Create for each sample an own folder
            sample_path = os.path.join(store_path, f"sample_{idx + self.offset}")
            os.makedirs(sample_path, exist_ok=True)

            # Save trajectory
            write(os.path.join(sample_path, "trajectory.xyz"), sample)
            
            # Save final sample
            write(os.path.join(sample_path, "final.xyz"), sample[-1])

    def sample(self, samples: List[ase.Atoms]):
        
        initial_rmsd = []
        final_rmsd = []
        for sample in samples:
            
            rxn = sample.info.get("rxn", None)
            if self.reference and rxn in self.reference:
                ref_atoms = self.reference[rxn]
                if len(ref_atoms) != len(sample):
                    raise ValueError(f"Reference and sample have different number of atoms for reaction {rxn}.")
                _, rmsd = get_rmsd(sample=sample, reference=ref_atoms, same_order=True)
                initial_rmsd.append(rmsd)
                
            batch = self.converter(sample)
            nfe, results = self.sampler(batch)
            
            # This is a list of iterations for all molecules at each iteration
            trajectory = [batch_inputs_to_atoms(result) for result in results]
            
            # This is a list of molecules at each iteration
            trajectory_atoms = []
            for i in range(len(trajectory[0])):
                trajectory_atoms.append([traj[i] for traj in trajectory])
            
            
            # Check the last atom in the trajectory
            _, rmsd = get_rmsd(sample=trajectory_atoms[0][-1], reference=ref_atoms, same_order=True)
            final_rmsd.append(rmsd)
            
            # Save the trajectory for each sample
            self.save_samples(
                trajectory_atoms,
                "trajectory",
                [],
                search_for_offset=True,
            )
            
        if len(initial_rmsd) > 0:
            print(
                f"Initial RMSD:\n Mean: {np.mean(initial_rmsd):.3f} \n "
                f"Median: {np.median(initial_rmsd):.3f}"
            )
            print(
                f"Final RMSD:\n Mean: {np.mean(final_rmsd):.3f} \n "
                f"Median: {np.median(final_rmsd):.3f}"
            )
            
            np.savez( 
                os.path.join( self.store_path, "rmsd"),
                rmsd_initial=rmsd_initial,
                rmsd_final=final_rmsd
            )

            
