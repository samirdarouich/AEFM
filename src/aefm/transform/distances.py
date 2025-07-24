from typing import Dict, Optional

import torch
import torch.nn as nn

from aefm import properties

class PairwiseDistancesCombined(nn.Module):
    """
    Compute pair-wise distances from indices provided by a neighbor list transform.
    """

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        if properties.reactant_coords in inputs:
            inputs = self.forward_one_graph(inputs, image_type=properties.IMAGE_TYPES["reactant"])
        if properties.product_coords in inputs:
            inputs = self.forward_one_graph(inputs, image_type=properties.IMAGE_TYPES["product"])
        
        return self.forward_one_graph(inputs)
        
    def forward_one_graph(self, inputs: Dict[str, torch.Tensor], image_type=None) -> Dict[str, torch.Tensor]:
        
        if image_type is not None:
            if image_type == properties.IMAGE_TYPES["reactant"]:
                R = inputs[properties.reactant_coords]
            elif image_type == properties.IMAGE_TYPES["product"]:
                R = inputs[properties.product_coords]
            else:
                raise ValueError(f"Unknown image type: {image_type}")
        else:
            R = inputs[properties.R]
        
        offsets = inputs[properties.offsets]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]

        # To avoid error in Windows OS
        idx_i = idx_i.long()
        idx_j = idx_j.long()

        Rij = R[idx_j] - R[idx_i] + offsets
        
        if image_type is not None:
            if image_type == properties.IMAGE_TYPES["reactant"]:
                inputs[properties.Rij_reactant] = Rij
            elif image_type == properties.IMAGE_TYPES["product"]:
                inputs[properties.Rij_product] = Rij
        else:
            inputs[properties.Rij] = Rij
        return inputs