from typing import Dict, List

import torch
from aefm import properties as structure

NOT_BATCHABLE = [structure.eig_energies, structure.eig_modes]

def _atoms_collate_fn(batch):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    elem = batch[0]
    idx_keys = {
        structure.idx_i,
        structure.idx_j,
        structure.idx_i_triples,
        structure.idx_i_fragment,
        structure.idx_j_fragment,
    }
    # Atom triple indices must be treated separately
    idx_triple_keys = {structure.idx_j_triples, structure.idx_k_triples}

    coll_batch = {}
    for key in elem:
        if (key not in idx_keys) and (key not in idx_triple_keys) and (
            key not in NOT_BATCHABLE):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch[structure.n_atoms], dim=0)
    seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch)), repeats=coll_batch[structure.n_atoms], dim=0
    )
    coll_batch[structure.idx_m] = idx_m

    if structure.conditions_idx_m in coll_batch:
        # Get the number of condition structures per batch
        conditions_idx_m = coll_batch[structure.conditions_idx_m].clone().detach()
        max_conditions = conditions_idx_m.max().item() + 1

        # This will ensure that each structure has a unique index (also for conditions)
        coll_batch[structure.conditions_idx_m + "_local"] = conditions_idx_m
        coll_batch[structure.conditions_idx_m] = (
            idx_m * max_conditions + conditions_idx_m
        )
        
        coll_batch[structure.conditions_n_atoms] = torch.bincount(coll_batch[structure.conditions_idx_m])
        

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in elem.keys():
            indices = []
            offset = 0
            for idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d[structure.idx_j].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch


def _reactions_collate_fn(
    batch: List[List[Dict[str, torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Build batch from reactions dataset, where each batch is a list of list of atoms

    Args:
        batch (List[List[Dict[str,torch.Tensor]]]): List of list of atoms given as
            Schnetopack compatible dictionaries of properties.

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    batch = [bb for b in batch for bb in b]
    return _atoms_collate_fn(batch)
