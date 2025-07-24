from typing import List, Optional, Tuple

import torch
from schnetpack.data import AtomsDataModule, AtomsLoader

from aefm.data import calculate_stats
from aefm.data.atoms import load_dataset
from aefm.data.loader import _reactions_collate_fn


class GDB7(AtomsDataModule):
    energy = "energy"  # ωB97X-D3/def2-TZVP: total energy

    property_unit_dict = {
        energy: "eV",
    }

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        group_by_reaction: bool = True,
        reaction_transforms: Optional[List[torch.nn.Module]] = None,
        persistent_workers: bool = False,
        **kwargs,
    ):
        """
        Initializes the Transition1x dataset with the given parameters.

        Args:
            datapath: Path to the dataset.
            batch_size: Number of samples per batch.
            group_by_reaction (bool): Whether to group data by reaction index or treat
                each entry individual.
            include_final_intermediates: Whether to include the final NEB intermediates
                from each reaction.
            include_all_intermediates: Whether to include all NEB intermediates from
                each reaction.
            reaction_transforms: List of transformations to apply to the reactions data.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        # Pass all relevant parameters to the parent class
        super().__init__(datapath=datapath, batch_size=batch_size, **kwargs)

        # Additional parameters
        self.reaction_transforms = reaction_transforms
        self.group_by_reaction = group_by_reaction
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        # check whether data needs to be copied
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            datapath = self._copy_to_workdir()

        # (re)load datasets
        if self.dataset is None:
            # Sort out intermediates
            self.dataset = load_dataset(
                datapath,
                self.format,
                property_units=self.property_units,
                distance_unit=self.distance_unit,
                load_properties=self.load_properties,
                reaction_transforms=self.reaction_transforms,
                group_by_reaction=self.group_by_reaction,
            )

            # load and generate partitions if needed
            if self.train_idx is None:
                self._load_partitions()

            # partition dataset
            self._train_dataset = self.dataset.subset(self.train_idx)
            self._val_dataset = self.dataset.subset(self.val_idx)
            self._test_dataset = self.dataset.subset(self.test_idx)

        self._setup_transforms()

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (property, divide_by_atoms, remove_atomref)
        if key in self._stats:
            return self._stats[key]

        stats = calculate_stats(
            self.train_dataloader(),
            divide_by_atoms={property: divide_by_atoms},
            atomref=self.train_dataset.atomrefs if remove_atomref else None,
        )[property]
        self._stats[key] = stats

        # Reset train dataloader in case of persistent workers
        self._train_dataloader = None
        return stats

    def train_dataloader(self) -> AtomsLoader:
        if self._train_dataloader is None:
            self._train_dataloader = AtomsLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self._pin_memory,
                collate_fn=_reactions_collate_fn,
                persistent_workers=self.persistent_workers,
            )
        return self._train_dataloader

    def val_dataloader(self) -> AtomsLoader:
        if self._val_dataloader is None:
            self._val_dataloader = AtomsLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_val_workers,
                pin_memory=self._pin_memory,
                collate_fn=_reactions_collate_fn,
                persistent_workers=self.persistent_workers,
            )
        return self._val_dataloader

    def test_dataloader(self) -> AtomsLoader:
        if self._test_dataloader is None:
            self._test_dataloader = AtomsLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                num_workers=self.num_test_workers,
                pin_memory=self._pin_memory,
                collate_fn=_reactions_collate_fn,
            )
        return self._test_dataloader
