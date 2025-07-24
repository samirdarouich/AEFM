from typing import List, Optional

import schnetpack as spk

__all__ = ["NeuralNetworkPotential"]


class NeuralNetworkPotential(spk.model.NeuralNetworkPotential):

    def __init__(self, additional_output_keys: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.additional_output_keys = additional_output_keys
        if additional_output_keys is not None:
            self.model_outputs += additional_output_keys
