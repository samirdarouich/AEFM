from typing import Dict, Final

from schnetpack.properties import *

IMAGE_TYPES: Dict[str, int] = {
    "reactant": 0,
    "product": 1,
    "transition_state": 2,
    "intermediate_final": 3,
    "intermediate": 4,
    "intermediate_normal_mode": 5,
    "intermediate_noise": 6,
    "transition_state_prior": 7,
    "gaussian_prior": 8,
}
IMAGE_TYPES_REVERSE: Dict[int, str] = {v: k for k, v in IMAGE_TYPES.items()}

reaction_ids: Final[str] = "_reaction_ids"
image_type: Final[str] = "_image_type"
x_0: Final[str] = "_x_0"
x_1: Final[str] = "_x_1"
conditions: Final[str] = "_conditions"
conditions_mask: Final[str] = f"{conditions}_mask"
conditions_idx_m: Final[str] = f"{conditions}_idx_m"
conditions_n_atoms: Final[str] = f"{conditions}_n_atoms"
subgraph_mask: Final[str] = "_subgraph_mask"
rmsd: Final[str] = "_rmsd"

reactant_coords: Final[str] = "_reactant_coords"
product_coords: Final[str] = "_product_coords"
Rij_reactant: Final[str] = "_Rij_reactant"
Rij_product: Final[str] = "_Rij_product"
idx_i_fragment: Final[str] = "_idx_i_fragment"
idx_j_fragment: Final[str] = "_idx_j_fragment"

eig_energies: Final[str] = "eig_energies"
eig_modes: Final[str] = "eig_modes"
nm_distplacement: Final[str] = "_nm_distplacement"
