import torch
import torchmetrics
from aefm import properties
from torch_scatter import scatter_mean


class CustomAccuracy(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with new batch of predictions and labels."""
        correct = (preds.argmax(dim=1) == target).sum()
        total = target.numel()
        self.correct += correct
        self.total += total

    def compute(self):
        """Compute final metric value."""
        return self.correct.float() / self.total
    
class RMSD:
    
    additional_target_properties = [properties.idx_m]
    
    def __call__(self, pred, target, idx_m):
        
        # Compute rmsd per molecule
        rmsd = scatter_mean(((pred - target)**2).sum(-1),idx_m,0)**0.5
        
        # Compute mean rmsd
        metric = rmsd.mean()
        return metric
    

class BondMAE:
    """
    Bond Mean Absolute Error (MAE) metric.
    This metric computes the mean absolute error between the predicted and target bond
    distances in a molecular structure. The error is calculated as the absolute
    difference between the predicted and target distance matrices. The metric is masked
    to consider only the bonds within the same molecule by using a block diagonal matrix.
    """
    additional_target_properties = [properties.n_atoms]

    def __call__(self, pred, target, n_atoms):
        
        # Compute distance matrix for both pred and target
        pred_cdist = torch.cdist(pred, pred)
        target_cdist = torch.cdist(target, target)
        
        # Create block diagonal matrix to mask out bonds between different molecules
        blocks = [torch.ones(n, n, device=target_cdist.device) for n in n_atoms]
        block_diag_matrix = torch.block_diag(*blocks)

        mae = torch.abs(pred_cdist - target_cdist)*block_diag_matrix
        metric = mae.sum() / block_diag_matrix.sum()
        return metric
    
class BondMAPE:
    """
    Bond Mean Absolute Percentage Error (MAPE) metric.
    This metric computes the mean absolute percentage error between the predicted 
    and target bond distances in a molecular structure. The error is calculated 
    as the absolute difference between the predicted and target distance matrices, 
    normalized by the target distances. The metric is masked to consider only 
    the bonds within the same molecule by using a block diagonal matrix.
    """
    additional_target_properties = [properties.n_atoms]

    def __call__(self, pred, target, n_atoms):
        
        # Compute distance matrix for both pred and target
        pred_cdist = torch.cdist(pred, pred)
        target_cdist = torch.cdist(target, target)
        
        # Create block diagonal matrix to mask out bonds between different molecules
        blocks = [torch.ones(n, n, device=target_cdist.device) for n in n_atoms]
        block_diag_matrix = torch.block_diag(*blocks)

        mape = torch.abs(pred_cdist - target_cdist)/(target_cdist+1e-3)*block_diag_matrix
        metric = mape.sum() / block_diag_matrix.sum()
        return metric