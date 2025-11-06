import numpy as np
from schnetpack.data import ASEAtomsData

import h5py
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

REFERENCE_ENERGIES = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
    9: -2712.8213146878606,
}


def get_molecular_reference_energy(atomic_numbers):
    molecular_reference_energy = 0
    for atomic_number in atomic_numbers:
        molecular_reference_energy += REFERENCE_ENERGIES[atomic_number]

    return molecular_reference_energy


def generator(formula, rxn, grp, skip_energy=[]):
    """Iterates through a h5 group"""
    energies = grp["wB97x_6-31G(d).energy"]
    forces = grp["wB97x_6-31G(d).forces"]
    atomic_numbers = list(grp["atomic_numbers"])
    positions = grp["positions"]
    molecular_reference_energy = get_molecular_reference_energy(atomic_numbers)

    for energy, force, positions in zip(energies, forces, positions):
        if energy in skip_energy:
            # print("skipping energy")
            continue
        d = {
            "rxn": rxn,
            "wB97x_6-31G(d).energy": energy.__float__(),
            "wB97x_6-31G(d).atomization_energy": energy
            - molecular_reference_energy.__float__(),
            "wB97x_6-31G(d).forces": force.tolist(),
            "positions": positions,
            "formula": formula,
            "atomic_numbers": atomic_numbers,
        }

        yield d


class Dataloader:
    """Can iterate through h5 data set for paper ####

    hdf5_file: path to data
    only_final: if True, the iterator will only loop through reactant, product and transition
    state instead of all configurations for each reaction and return them in dictionaries.
    """

    def __init__(self, hdf5_file, datasplit="data", only_final=False):
        self.hdf5_file = hdf5_file
        self.only_final = only_final

        self.datasplit = datasplit
        if datasplit:
            assert datasplit in [
                "data",
                "train",
                "val",
                "test",
            ], "datasplit must be one of 'all', 'train', 'val' or 'test'"

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]

            for formula, grp in split.items():  # type: ignore
                for rxn, subgrp in grp.items():
                    reactant = next(generator(formula, rxn, subgrp["reactant"]))
                    product = next(generator(formula, rxn, subgrp["product"]))
                    transition_state = next(
                        generator(formula, rxn, subgrp["transition_state"])
                    )

                    if self.only_final:
                        yield {
                            "rxn": rxn,
                            "reactant": reactant,
                            "product": product,
                            "transition_state": transition_state,
                        }
                    else:
                        skip_energy = [
                            reactant["wB97x_6-31G(d).energy"],
                            product["wB97x_6-31G(d).energy"],
                        ]
                        yield {
                            "rxn": rxn,
                            "reactant": reactant,
                            "product": product,
                            "transition_state": transition_state,
                            "intermediates": list(
                                generator(formula, rxn, subgrp, skip_energy=skip_energy)
                            ),
                        }


def get_atoms(configuration):
    atoms = Atoms(configuration["atomic_numbers"])
    centroid_pos = configuration["positions"] - np.mean(
        configuration["positions"], axis=0
    )
    atoms.set_positions(centroid_pos)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=configuration["wB97x_6-31G(d).energy"],
        forces=configuration["wB97x_6-31G(d).forces"],
    )

    return atoms


ONLY_RTSP = True

dataloader = Dataloader("Transition1x.h5", datasplit="data", only_final=ONLY_RTSP)

atoms_list = []

for configuration in dataloader:
    rxn = int(configuration["rxn"][3:])
    reactant = get_atoms(configuration["reactant"])
    reactant.info.update({"rxn": rxn, "type": "reactant"})
    product = get_atoms(configuration["product"])
    product.info.update({"rxn": rxn, "type": "product"})
    transition_state = get_atoms(configuration["transition_state"])
    transition_state.info.update({"rxn": rxn, "type": "transition_state"})

    # If not only reactant, product and transition state, NEB intermediates are added
    if not ONLY_RTSP:
        # 10 images in total (including reactant and product), thus 8 intermediates
        # including transition state. last 8 images are final path
        assert len(configuration["intermediates"]) % 8 == 0, "NEB path has wrong length"

        intermediates_final = [
            get_atoms(conf) for conf in configuration["intermediates"][-8:]
        ]
        intermediates = [get_atoms(conf) for conf in configuration["intermediates"][:-8]]

        flag = False
        for intermediate in intermediates_final:
            if (
                intermediate.get_potential_energy()
                == transition_state.get_potential_energy()
            ):
                flag = True
                intermediate.info.update(
                    {"rxn": rxn, "type": "transition_state"}
                )
            else:
                intermediate.info.update(
                    {"rxn": rxn, "type": "intermediate_final"}
                )
        assert flag, "Transition state not found in final path"

        for intermediate in intermediates:
            intermediate.info.update({"rxn": rxn, "type": "intermediate"})
        path = [reactant] + intermediates + intermediates_final + [product]

    else:
        path = [reactant, transition_state, product]
    
    atoms_list.extend(path)

# Sort after reaction index
atoms_list_sorted = sorted(atoms_list, key=lambda x: x.info["rxn"])

# Write all data to extxyz
write("t1x.xyz", atoms_list_sorted)

# Write only TS to extxyz
ts_atoms = [a for a in atoms_list_sorted if a.info["type"] == "transition_state"]
write("t1x_ts.xyz", ts_atoms)

