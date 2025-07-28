import argparse
import shutil

import pandas as pd
from ase import Atoms
from ase.build.rotate import minimize_rotation_and_translation
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read, write
import os

def _compute_rmsd(atom1: Atoms, atom2: Atoms):
    """
    Compute the RMSD between two atomic structures.
    (https://en.wikipedia.org/wiki/Root_mean_square_deviation_of_atomic_positions)

    RMSD(v,w) = sqrt( 1/n sum_i^n ||v_i-w_i||^2 )
              = sqrt( 1/n sum_i^n ( (v_i,x-w_i,x)^2 + (v_i,y-w_i,y)^2 + (v_i,z-w_i,z)^2 )

    """
    if sorted(atom1.get_atomic_numbers()) != sorted(atom2.get_atomic_numbers()):
        raise ValueError("The number of the same species aren't matching!")
    return ((atom1.positions - atom2.positions) ** 2).sum(-1).mean() ** 0.5


parser = argparse.ArgumentParser(
    description="Evaluate RMSD and delta E for generated structures."
)
parser.add_argument(
    "--ncores",
    type=str,
    help="Number of cores to use for ORCA calculations.",
)
parser.add_argument(
    "--data_version",
    type=str,
    help="Folder of the non-equilibrium database.",
)
parser.add_argument(
    "--data_source",
    type=str,
    help="Path to the non-equilibrium database xyz file.",
)
args = parser.parse_args()
args.ncores = int(args.ncores)

print(f"Using {args.ncores} cores for ORCA calculations.")

# Setup calculator
profile = OrcaProfile(command="/opt/bwhpc/common/chem/orca/5.0.4_static/orca")

calc = ORCA(
    profile=profile,
    charge=0,
    mult=1,
    orcasimpleinput="wb97x 6-31G(d) Engrad",
    orcablocks=f"%pal nprocs {args.ncores} end",
    directory="orca",
)


database = read(
    "datasets/t1x/t1x.xyz", "1::3",
)

datasource = args.data_source
samples = read(
    f"data/{args.data_version}/{datasource}.xyz",
    ":",
)
labeled_db = (
    f"data/{args.data_version}/{datasource}_dft_labeled.xyz"
)
print(
    f"Found {len(samples)} samples to analyze in {args.data_version} for {datasource}."
)

rmsds_delta_e = {
    "rmsd": {},
    "delta_e": {},
}

for atoms in samples:
    # Get reference
    reaction_id = atoms.info["rxn"]

    print(f"\nAnalyzing reaction '{reaction_id}'")

    reference = database[reaction_id]

    assert (
        (int(reference.info["rxn"][3:]) == reaction_id)
        and (reference.info["type"] == "transition_state")
        and len(reference) == len(atoms)
    ), "Wrong reference"

    # Get align and get rmsd
    minimize_rotation_and_translation(reference, atoms)
    rmsd = _compute_rmsd(reference, atoms)
    atoms.info["rmsd"] = rmsd
    atoms.info["rxn"] = reaction_id
    rmsds_delta_e["rmsd"][reaction_id] = rmsd

    # Get delta E in eV
    atoms.calc = calc
    try:
        delta_e = reference.get_potential_energy() - atoms.get_potential_energy()
    except Exception as e:
        print(f"Error during energy calculation for reaction {reaction_id}: {e}")
        delta_e = float("inf")
    write(labeled_db, atoms, append=True)
    rmsds_delta_e["delta_e"][reaction_id] = delta_e
    shutil.rmtree("orca", ignore_errors=True)

    print(f"RMSD: {rmsd:.2f} Å, Delta E: {delta_e:.2f} eV")

print("Saving results to csv file.")
csv_path = (
    f"csv/{args.data_version}"
)
os.makedirs(csv_path, exist_ok=True)
df = pd.DataFrame(rmsds_delta_e)
df.to_csv(f"{csv_path}/sample_{datasource}.csv")
print(df.abs().mean())
