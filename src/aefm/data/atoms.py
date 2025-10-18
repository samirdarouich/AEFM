import logging
from typing import Dict, List, Optional

import torch
from schnetpack.data import (
    ASEAtomsData,
    AtomsDataFormat,
    BaseAtomsData,
)
from schnetpack.data.atoms import AtomsDataError
from schnetpack.transform import Transform

import aefm.properties as structure

logger = logging.getLogger(__name__)

__all__ = ["ASEReactionData"]


class ASEReactionData(ASEAtomsData):
    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[torch.nn.Module]] = None,
        reaction_transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        reaction_id_key: str = "reaction_ids_unique",
        group_by_reaction: bool = True,
        include_final_intermediates: bool = False,
        include_all_intermediates: bool = False,
    ):
        """
        Initializes the dataset with the given parameters.
        Args:
            datapath: Path to ASE DB.
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_structure: If True, load structure properties.
            transforms: preprocessing torch.nn.Module (see schnetpack.data.transforms)
            reaction_transforms: List of transformations to apply to the reactions.
            subset_idx: List of data indices.
            property_units: unit string dictionary that overwrites the native units of
                the dataset. Units are converted automatically during loading.
            distance_unit: Unit of distance. If None, the unit is read from the ASE db.
            group_by_reaction (bool): Whether to group data by reaction index or treat
                each entry individual. 
            include_final_intermediates (bool): Whether to include final intermediates.
            include_all_intermediates (bool): Whether to include all intermediates. 
        """
        super().__init__(
            datapath=datapath,
            load_properties=load_properties,
            load_structure=load_structure,
            transforms=transforms,
            subset_idx=subset_idx,
            property_units=property_units,
            distance_unit=distance_unit,
        )

        self.reaction_transforms = reaction_transforms  # type: ignore
        self.reaction_id_key = reaction_id_key
        self.group_by_reaction = group_by_reaction
        self.include_final_intermediates = include_final_intermediates
        self.include_all_intermediates = include_all_intermediates

        # Cache reaction meta data
        self._cache_reaction_metadata()

    @property
    def reaction_transforms(self):
        return self._reaction_transforms

    @reaction_transforms.setter
    def reaction_transforms(self, value: Optional[List[Transform]]):
        self._reaction_transforms = []
        self._reaction_transform_module = None

        if value is not None:
            for tf in value:
                self._reaction_transforms.append(tf)
            self._reaction_transform_module = torch.nn.Sequential(
                *self._reaction_transforms
            )

    def _apply_reaction_transforms(self, props):
        if self._reaction_transform_module is not None:
            props = self._reaction_transform_module(props)
        return props

    def _cache_reaction_metadata(self):
        """Precompute and cache the meta data of the reactions."""
        self._reaction_indices = {}
        self._reaction_ids = []
        self._image_types = []
        for i, (ridx, itype) in enumerate(
            zip(
                self.metadata["groups_ids"][self.reaction_id_key],
                self.metadata["groups_ids"]["image_type"],
            )
        ):
            self._reaction_ids.append(ridx)
            self._image_types.append(itype)

            # Either group by reaction or keep all images separate
            if not self.group_by_reaction:
                self._reaction_indices[i] = [i]
            else:
                if ridx not in self._reaction_indices:
                    self._reaction_indices[ridx] = []

                # Don't use (final) intermediates
                if not self.include_all_intermediates:
                    if (
                        itype == "intermediate_final"
                        and not self.include_final_intermediates
                    ):
                        continue
                    if itype == "intermediate":
                        continue

                self._reaction_indices[ridx].append(i)

    def __len__(self) -> int:
        if self.subset_idx is not None:
            return len(self.subset_idx)

        return len(self._reaction_indices)

    def __getitem__(self, idx: int) -> List[Dict[str, torch.Tensor]]:
        if self.subset_idx is not None:
            idx = self.subset_idx[idx]

        reaction_indices = self._reaction_indices[idx]

        # Get the reaction images
        images = [
            self._get_properties(
                self.conn, i, self.load_properties, self.load_structure
            )
            for i in reaction_indices
        ]

        # Apply reaction transforms
        images = self._apply_reaction_transforms(images)

        # Apply the individual transforms to the output
        images = [self._apply_transforms(image) for image in images]

        return images

    def _get_properties(
        self, conn, idx: int, load_properties: List[str], load_structure: bool
    ):
        row = conn.get(idx + 1)

        # extract properties
        # TODO: can the copies be avoided?
        properties = {}
        properties[structure.reaction_ids] = torch.tensor([self._reaction_ids[idx]])
        properties[structure.image_type] = torch.tensor(
            [structure.IMAGE_TYPES[self._image_types[idx]]]
        )
        properties[structure.idx] = torch.tensor([idx])
        for pname in load_properties:
            properties[pname] = (
                torch.tensor(row.data[pname].copy()) * self.conversions[pname]
            )

        Z = row["numbers"].copy()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]])

        if load_structure:
            properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
            properties[structure.position] = (
                torch.tensor(row["positions"].copy()) * self.distance_conversion
            )
            properties[structure.cell] = (
                torch.tensor(row["cell"][None].copy()) * self.distance_conversion
            )
            properties[structure.pbc] = torch.tensor(row["pbc"])

        return properties

    @property
    def unique_reactions_ids(self) -> List[int]:
        reaction_indices = list(self._reaction_indices.keys())
        if self.subset_idx is not None:
            return [reaction_indices[i] for i in self.subset_idx]
        return reaction_indices

    @property
    def n_images(self) -> int:
        return len(next(iter(self._reaction_indices.values()), []))


def load_dataset(datapath: str, format: AtomsDataFormat, **kwargs) -> BaseAtomsData:
    """
    Load dataset.

    Args:
        datapath: file path
        format: atoms data format
        **kwargs: arguments for passed to AtomsData init

    """
    if format is AtomsDataFormat.ASE:
        dataset = ASEReactionData(datapath=datapath, **kwargs)
    else:
        raise AtomsDataError(f"Unknown format: {format}")
    return dataset
