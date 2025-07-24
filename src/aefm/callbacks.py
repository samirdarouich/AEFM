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
from aefm.utils import batch_inputs_to_atoms, batch_rmsd


class SamplerCallback(Callback):
    """
    Callback to sample molecules during validation or testing.
    """

    def __init__(
        self,
        sampler: Sampler,
        name: str = "sampling",
        store_path: str = "samples",
        every_n_batchs_train: int = 1,
        every_n_epochs_train: int = 1,
        start_epoch_train: int = 1,
        every_n_batchs_val: int = 1,
        every_n_epochs_val: int = 1,
        start_epoch_val: int = 1,
        log_rmsd: bool = True,
        same_order: bool = True,
        max_permutations: int = 1e6,
        ignore_chirality: bool = False,
        log_trajectory: bool = False,
        t_trajectory: Optional[List[float] | int] = None,
    ):
        """
        Args:
            sampler: sampler to be used for sampling.
            name: name of the callback.
            store_path: path to store the results and samples.
            every_n_batchs: sample every n batches for validation.
            every_n_epochs: sample every n epochs for validation.
            start_epoch: start sampling at this epoch for validation.
            log_rmsd: whether to log the RMSD of sampled structures. Useful for
                relaxation tasks of the atoms positions R.
            same_order: whether atoms of sample and target have the same ordering.
            max_permutations: Maximum number of permutations to consider for the
                alignment of the atoms. If the number of permutations is larger than
                this value, the alignment is skipped.
            ignore_chirality: If True, the function also calculates RMSD for the
                molecule reflected along the z-axis, and returns the minimum structure
                and RMSD.
            log_trajectory: whether to log the trajectory of the sampling process.
            t_trajectory: time steps (list of floats or number for linspace) for which
                the trajectory should be logged.
        """
        super().__init__()
        self.sampler = sampler
        self.name = name
        self.store_path = store_path
        self.every_n_batchs_train = every_n_batchs_train
        self.every_n_epochs_train = every_n_epochs_train
        self.start_epoch_train = start_epoch_train
        self.every_n_batchs_val = every_n_batchs_val
        self.every_n_epochs_val = every_n_epochs_val
        self.start_epoch_val = start_epoch_val
        self.log_rmsd = log_rmsd
        self.same_order = same_order
        self.max_permutations = max_permutations
        self.ignore_chirality = ignore_chirality
        self.log_trajectory = log_trajectory

        if log_trajectory and t_trajectory is not None:
            raise ValueError("t_trajectory must be provided if log_trajectory is True.")

        # Define trajectory for samples
        if t_trajectory is None:
            self.t_trajectory = torch.tensor([0.0, 1.0])
        else:
            if isinstance(t_trajectory, int):
                self.t_trajectory = torch.linspace(0.0, 1.0, t_trajectory)
            else:
                self.t_trajectory = torch.tensor(t_trajectory)

        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)

        # Define offset for saving samples
        self.offset = 0

    def sample(
        self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs
    ) -> Tuple[int, List[Dict[str, torch.Tensor]]]:
        """
        Samples or denoises a batch of molecules.

        Args:
            model: model to sample from.
            batch: input batch to be used for sampling/denoising.

        Returns:
            number of function evaluations, sampled data as list of dictionaries.
        """

        # update the sampling model
        self.sampler.update_model(model)

        # sample (whole trajectory, where the time is dimension 0 in increasing order)
        nfe, trajectory = self.sampler(batch, t=self.t_trajectory, **kwargs)

        # restore model
        self.sampler.restore_model()

        return nfe, trajectory

    def save_samples(
        self,
        samples: Sequence[Union[Atoms, List[Atoms]]],
        epoch: int,
        keyword: str,
        path_keywords: List[str],
        test=False,
        search_for_offset=True,
    ):
        """
        Saves the samples to disk.

        Args:
            samples: List of ASE Atoms of the final samples to be saved.
            epoch: current epoch.
            batch_idx: current batch index.
            keyword: keyword to be used for the file name.
            path_keywords: List of keywords to be used for the path.
            test: whether the samples are from the test set.
            search_for_offset: whether to search for an offset in the sample folder
                names. If not it will take the class attribute offset.
        """
        phase = "test" if test else "val"

        # Define store path
        store_path = os.path.join(
            self.store_path, f"phase_{phase}", f"epoch_{epoch}", *path_keywords
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

            sample_path = os.path.join(sample_path, f"sample_{keyword}.xyz")
            write(sample_path, sample)

    def _step(
        self,
        pl_module: LightningModule,
        batch: Dict[str, torch.Tensor],
        phase: str,
    ):
        """
        Perform a sampling step and log results.

        Args:
            model: model to sample from.
            batch: input batch to be used for sampling/denoising.
            phase: train, val, test
        """
        test = phase == "test"

        # Check if conditions are also in target (joint modeling)
        conditions_in_target = False
        if properties.x_0 in batch:
            conditions_in_target = (
                batch[properties.x_0].shape[0] == batch[properties.R].shape[0]
            )

        # sample a batch
        nfe, results = self.sample(
            pl_module.model, batch, conditions_in_target=conditions_in_target
        )

        # log the trajectory of the sampling process
        if self.log_trajectory and not test:
            # This is a list timesteps for each molecule
            trajectory = [batch_inputs_to_atoms(result) for result in results]

            # This is a list of molecules at each timestep
            trajectory_atoms = []
            for i in range(len(results)):
                trajectory_atoms.append([traj[i] for traj in trajectory])

            # Save the trajectory for each sample
            self.save_samples(
                trajectory_atoms,
                pl_module.trainer.current_epoch,
                "trajectory",
                [f"nfe_{nfe}"],
                test=test,
                search_for_offset=True,
            )

        metrics = {}
        # compute RMSD of sampled structures if requested
        if self.log_rmsd:
            # get the reference positions
            reference_R = (
                batch[f"target_{properties.R}"]
                if f"target_{properties.R}" in batch
                else batch[properties.R]
            )

            # get rmsd of final samples
            samples = results[-1]
            aligned_mols, generated_mols, rmsds = batch_rmsd(
                reference_R,
                samples,
                self.same_order,
                self.ignore_chirality,
                self.max_permutations,
            )

            metrics["rmsd_mean"] = rmsds.mean()
            metrics["rmsd_median"] = rmsds.median()

            # get rmsd of initial samples
            samples = results[0]
            _, _, rmsds = batch_rmsd(
                reference_R,
                samples,
                self.same_order,
                self.ignore_chirality,
                self.max_permutations,
            )

            metrics["rmsd_x_0_mean"] = rmsds.mean()
            metrics["rmsd_x_0_median"] = rmsds.median()

            # save the generated molecules (take offset from trajectory if available)
            if test:
                self.save_samples(
                    generated_mols,
                    pl_module.trainer.current_epoch,
                    "generated",
                    [f"nfe_{nfe}"],
                    test=test,
                    search_for_offset=False if self.log_trajectory else True,
                )

                # save the aligned molecules
                self.save_samples(
                    aligned_mols,
                    pl_module.trainer.current_epoch,
                    "aligned",
                    [f"nfe_{nfe}"],
                    test=test,
                    search_for_offset=False,
                )

        # log the metrics
        if metrics:
            for key, val in metrics.items():
                pl_module.log(
                    f"{phase}/{self.name}_{key}",
                    val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=int(batch[properties.idx_m].max().item() + 1),
                )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Overwrites ``on_validation_batch_end`` hook from ``Callback``.
        """
        # sample only every n batches and m epochs
        if (
            trainer.current_epoch >= self.start_epoch_train
            and trainer.current_epoch % self.every_n_epochs_train == 0
            and batch_idx % self.every_n_batchs_train == 0
        ):
            self._step(pl_module, batch, phase="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Overwrites ``on_validation_batch_end`` hook from ``Callback``.
        """
        # sample only every n batches and m epochs
        if (
            trainer.current_epoch >= self.start_epoch_val
            and trainer.current_epoch % self.every_n_epochs_val == 0
            and batch_idx % self.every_n_batchs_val == 0
        ):
            self._step(pl_module, batch, phase="val")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Overwrites ``on_test_batch_end`` hook from ``Callback``.
        """
        self._step(pl_module, batch, phase="test")


class SamplerFlowDiffCallback(SamplerCallback):

    def sample(
        self,
        model: nn.Module,
        denoiser: nn.Module,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Tuple[int, List[Dict[str, torch.Tensor]]]:
        """
        Samples or denoises a batch of molecules.

        Args:
            model: model to sample from.
            batch: input batch to be used for sampling/denoising.

        Returns:
            number of function evaluations, sampled data as list of dictionaries.
        """

        # update the sampling model
        self.sampler.update_model(model, denoiser)

        # sample (whole trajectory, where the time is dimension 0 in increasing order)
        nfe, trajectory = self.sampler(batch, t=self.t_trajectory, **kwargs)

        # restore model
        self.sampler.restore_model()

        return nfe, trajectory

    def _step(
        self,
        pl_module: LightningModule,
        batch: Dict[str, torch.Tensor],
        phase: str,
    ):
        """
        Perform a sampling step and log results.

        Args:
            model: model to sample from.
            batch: input batch to be used for sampling/denoising.
            phase: train, val, test
        """
        test = phase == "test"

        # Check if conditions are also in target (joint modeling)
        conditions_in_target = False
        if properties.x_0 in batch:
            conditions_in_target = (
                batch[properties.x_0].shape[0] == batch[properties.R].shape[0]
            )

        # sample a batch
        nfe, results = self.sample(
            pl_module.model,
            pl_module.denoiser_net,
            batch,
            conditions_in_target=conditions_in_target,
        )

        # log the trajectory of the sampling process
        if self.log_trajectory and not test:
            # This is a list timesteps for each molecule
            trajectory = [batch_inputs_to_atoms(result) for result in results]

            # This is a list of molecules at each timestep
            trajectory_atoms = []
            for i in range(len(results)):
                trajectory_atoms.append([traj[i] for traj in trajectory])

            # Save the trajectory for each sample
            self.save_samples(
                trajectory_atoms,
                pl_module.trainer.current_epoch,
                "trajectory",
                [f"nfe_{nfe}"],
                test=test,
                search_for_offset=True,
            )

        metrics = {}
        # compute RMSD of sampled structures if requested
        if self.log_rmsd:
            # get the reference positions
            reference_R = (
                batch[f"target_{properties.R}"]
                if f"target_{properties.R}" in batch
                else batch[properties.R]
            )

            # get rmsd of final samples
            samples = results[-1]
            aligned_mols, generated_mols, rmsds = batch_rmsd(
                reference_R,
                samples,
                self.same_order,
                self.ignore_chirality,
                self.max_permutations,
            )

            metrics["rmsd_mean"] = rmsds.mean()
            metrics["rmsd_median"] = rmsds.median()

            # get rmsd of initial samples
            samples = results[0]
            _, _, rmsds = batch_rmsd(
                reference_R,
                samples,
                self.same_order,
                self.ignore_chirality,
                self.max_permutations,
            )

            metrics["rmsd_x_0_mean"] = rmsds.mean()
            metrics["rmsd_x_0_median"] = rmsds.median()

            # save the generated molecules (take offset from trajectory if available)
            if test:
                self.save_samples(
                    generated_mols,
                    pl_module.trainer.current_epoch,
                    "generated",
                    [f"nfe_{nfe}"],
                    test=test,
                    search_for_offset=False if self.log_trajectory else True,
                )

                # save the aligned molecules
                self.save_samples(
                    aligned_mols,
                    pl_module.trainer.current_epoch,
                    "aligned",
                    [f"nfe_{nfe}"],
                    test=test,
                    search_for_offset=False,
                )

        # log the metrics
        if metrics:
            for key, val in metrics.items():
                pl_module.log(
                    f"{phase}/{self.name}_{key}",
                    val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=int(batch[properties.idx_m].max().item() + 1),
                )
