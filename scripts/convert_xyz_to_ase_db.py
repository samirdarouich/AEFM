import argparse
from pathlib import Path

import numpy as np
from ase.io import read
from schnetpack.data import ASEAtomsData
from tqdm import tqdm
from aefm.utils import pymatgen_align

parser = argparse.ArgumentParser(description="Convert extended XYZ to an ASE database.")
parser.add_argument(
    "--data_path",
    help="Path to input extended xyz file (or pattern) to read with ase.io.read. Assumes R, TS, P for each reaction in this order.",
)
parser.add_argument("--output_path", help="Path to output ASE database file")
parser.add_argument(
    "--unlabeled",
    dest="labeled_dataset",
    action="store_false",
    help="Create an unlabeled dataset (no energies/forces). By default the dataset is labeled.",
)
parser.add_argument(
    "--meta-keys",
    dest="meta_keys",
    type=lambda s: [k.strip() for k in s.split(",") if k.strip()],
    default=[],
    help=(
        "Comma-separated list of metadata keys to extract from each ASE Atoms object. "
        "Keys may refer to atoms.info entries or common Atoms attributes (cell, pbc, tags, magmoms, positions, constraints, calculator). "
        "Default: ''"
    ),
)


args = parser.parse_args()

data_path = args.data_path
output_path = args.output_path
labeled_dataset = args.labeled_dataset
meta_keys = args.meta_keys

txt = f"""
Converting extended XYZ data at '{data_path}' to ASE database at '{output_path}'.
Expecting R, TS, P for each reaction in this order. 
Aligns product to reactant and removes center of positions (COP).
"""
print(txt)

if labeled_dataset:
    print("Extracting energies and forces.")
    
if meta_keys:
    print(f"Extracting metadata keys: {meta_keys}")

# Ensure output directory exists
out_dir = Path(output_path).parent
if out_dir and not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=True)


database = read(data_path, index=":")
assert len(database) % 3 == 0, "Database length is not multiple of 3 (R, TS, P)"

print(f"Total reactions in input data: {len(database) // 3}")

reactants, transition_states, products = database[0::3], database[1::3], database[2::3]

properties = []
atoms_list = []
reaction_ids = []
reaction_ids_unique = []
type_list = []
formula_list = []
meta_data = {key: [] for key in meta_keys}

for i in tqdm(range(len(reactants)), desc="Processing reactions"):
    reactant = reactants[i]
    transition_state = transition_states[i]
    product = products[i]

    rxn = int(reactant.info.get("rxn", i))
    rxn_unique = i
    formula = reactant.get_chemical_formula()

    reactant.info.update({"type": "reactant"})
    transition_state.info.update({"type": "transition_state"})
    product.info.update({"type": "product"})

    path = [reactant, transition_state, product]

    if labeled_dataset:
        properties.extend(
            {"energy": np.array([p.get_potential_energy()]), "forces": p.get_forces()}
            for p in path
        )
    else:
        properties.extend({} for p in path)
    
    # Remove COP
    for p in path:
        p.positions = p.positions - p.positions.mean(axis=0)

    # Align reactant and product
    path[2] = pymatgen_align(path[2], path[1], same_order=True)

    atoms_list.extend(path)
    type_list.extend([p.info["type"] for p in path])
    reaction_ids.extend([rxn for _ in path])
    reaction_ids_unique.extend([rxn_unique for _ in path])
    formula_list.extend([formula for _ in path])

    for key in meta_keys:
        for p in path:
            if key in p.info:
                meta_data[key].append(str(p.info[key]))
            elif hasattr(p, key):
                meta_data[key].append(str(getattr(p, key)))
            else:
                raise KeyError(
                    f"Metadata key '{key}' not found in atoms.info or as Atoms attribute."
                )

# Sort after reaction index
sorting_idx = np.argsort(reaction_ids, stable=True)
properties_sorted = [properties[i] for i in sorting_idx]
atoms_list_sorted = [atoms_list[i] for i in sorting_idx]
type_list_sorted = [type_list[i] for i in sorting_idx]
reaction_ids_sorted = [reaction_ids[i] for i in sorting_idx]
reaction_ids_unique_sorted = [reaction_ids_unique[i] for i in sorting_idx]
formula_list_sorted = [formula_list[i] for i in sorting_idx]
meta_data_sorted = {key: [meta_data[key][i] for i in sorting_idx] for key in meta_keys}

### Write full dataset to ASE database ###
# Write to ASE db
property_unit_dict = {}
if labeled_dataset:
    property_unit_dict = {"energy": "eV", "forces": "eV/Ang"}

print(f"Writing dataset to {output_path}...")
dataset = ASEAtomsData.create(
    output_path,
    distance_unit="Ang",
    property_unit_dict=property_unit_dict,
)
dataset.add_systems(property_list=properties_sorted, atoms_list=atoms_list_sorted)
dataset.update_metadata(
    groups_ids={
        "reaction_ids": reaction_ids_sorted,
        "reaction_ids_unique": reaction_ids_unique_sorted,
        "image_type": type_list_sorted,
        "formula": formula_list_sorted,
        **meta_data_sorted,
    }
)
print("Done.")
