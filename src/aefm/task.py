import logging
from typing import Dict, Optional

import torch
from schnetpack.task import (
    AtomisticTask,
    ModelOutput,
    UnsupervisedModelOutput,
)
from torch import nn

from aefm import properties

log = logging.getLogger(__name__)

__all__ = [
    "NLLOutput",
    "BondModelOutput",
    "GenerativeTask",
    "ConsiderOnlySelectedAtoms",
]


class NLLOutput(ModelOutput):
    """ NLL training output to allow learning of a shallow ensemble."""
    def calculate_loss(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        calculate the loss.

        Args:
            pred: outputs.
            target: target values.
        """
        eps = 1e-6
        mean = pred[self.name]
        target = target[self.target_property]
        sigmas = pred[self.name + "_uncertainty"]
        variances = torch.pow(sigmas, 2)
        # Ensure numerical stability
        variances = torch.clip(variances, eps)

        l1 = torch.log(variances)
        l2 = ((mean - target) ** 2) / variances
        nll = 0.5 * (l1 + l2)

        loss = self.loss_weight * nll.mean()
        return loss


class BondModelOutput(ModelOutput):
    """ Physical bond loss to avoid artefacts in generative models."""
    additional_target_properties = [properties.idx_i, properties.idx_j]
    
    def __init__(self, cutoff: float = None, include_pred: bool = False, remove_conditions: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.include_pred = include_pred
        self.remove_conditions = remove_conditions
        self.name = self.name + "_distances"
        self.target_property = self.target_property + "_distances"

    def calculate_loss(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:

        # Extract distances
        target_distances = target[self.target_property]
        pred_distances = pred[self.name]

        if properties.conditions_mask in target:
            # In case conditioning structures are present and should not be modeled in
            # the loss, remove these edges.
            if self.remove_conditions:
                # get original edge indices
                idx_i = target[properties.idx_i]
                idx_j = target[properties.idx_j]
                
                # select only edges between non-condition atoms
                considered_atoms = target[properties.conditions_mask].nonzero()[:, 0]
                mask = torch.isin(idx_i, considered_atoms) & torch.isin(idx_j, considered_atoms)
                target_distances = target_distances[mask]
                pred_distances = pred_distances[mask]

            # In case conditioning structures are present but should be modeled, only
            # consider distances within each structure but not between different
            # structures
            elif target[properties.subgraph_mask] is not None:
                subgraph_mask = target[properties.subgraph_mask].squeeze().bool()
                target_distances = target_distances[subgraph_mask]
                pred_distances = pred_distances[subgraph_mask]

        # Mask out distances above cutoff (based on the target distances)
        if self.cutoff is not None and self.cutoff > 0:
            mask = target_distances < self.cutoff
            if self.include_pred:
                mask = mask | (pred_distances < self.cutoff)
            target_distances = target_distances[mask]
            pred_distances = pred_distances[mask]

        # Compute loss
        loss = self.loss_weight * self.loss_fn(pred_distances, target_distances)

        return loss

    def update_metrics(self, pred, target, subset):

        # Extract distances
        target_distances = target[self.target_property]
        pred_distances = pred[self.name]

        if properties.conditions_mask in target:
            # In case conditioning structures are present and should not be modeled in
            # the loss, remove these edges.
            if self.remove_conditions:
                # get original edge indices
                idx_i = target[properties.idx_i]
                idx_j = target[properties.idx_j]
                
                # select only edges between non-condition atoms
                considered_atoms = target[properties.conditions_mask].nonzero()[:, 0]
                mask = torch.isin(idx_i, considered_atoms) & torch.isin(idx_j, considered_atoms)
                target_distances = target_distances[mask]
                pred_distances = pred_distances[mask]

            # In case conditioning structures are present but should be modeled, only
            # consider distances within each structure but not between different
            # structures
            elif target[properties.subgraph_mask] is not None:
                subgraph_mask = target[properties.subgraph_mask].squeeze().bool()
                target_distances = target_distances[subgraph_mask]
                pred_distances = pred_distances[subgraph_mask]

        # Mask out distances above cutoff
        if self.cutoff is not None and self.cutoff > 0:
            mask = target_distances < self.cutoff
            if self.include_pred:
                mask = mask | (pred_distances < self.cutoff)
            target_distances = target_distances[mask]
            pred_distances = pred_distances[mask]

        for metric in self.metrics[subset].values():
            metric(pred_distances, target_distances)


class GenerativeTask(AtomisticTask):
    """
    Defines the generative models task for pytorch lightning.
    Subclasses the atomistic task and adds the flow/diffusion.
    """

    def __init__(
        self,
        skip_exploding_batches: bool = True,
        **kwargs,
    ):
        """
        Args:
            skip_exploding_batches: ignore exploding batches during training.
        """
        super().__init__(**kwargs)
        self.skip_exploding_batches = skip_exploding_batches

    def setup(self, stage=None):
        """
        overwrite the pytorch lightning task setup function.
        """
        # call the parent atomistic task setup
        AtomisticTask.setup(self, stage=stage)  # type: ignore

        # force some post-processing transforms during training
        forced_postprocessors = []
        for pp in self.model.postprocessors:
            if hasattr(pp, "force_apply"):
                if pp.force_apply:
                    forced_postprocessors.append(pp)
        self.model.forced_postprocessors = nn.ModuleList(forced_postprocessors)

    def predict_without_postprocessing(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        predict without post-processing transforms.
        Note: forced post-processing transforms will still be applied.

        Args:
            batch: input batch.
        """
        tmp_postprocessors = self.model.postprocessors
        self.model.postprocessors = self.model.forced_postprocessors
        pred = self(batch)
        self.model.postprocessors = tmp_postprocessors

        return pred

    def log_metrics(self, pred, targets, subset, batch_size):
        for output in self.outputs:
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}/{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                    batch_size=batch_size,
                )
    
    def _step(self, batch: Dict[str, torch.Tensor], subset: str) -> torch.FloatTensor:
        """
        perform one forward pass and calculate the loss and log metrics.

        Args:
            batch: input batch.
            subset: the dataset split used.
        """
        # extract the target values from the batch
        targets = {}
        extract_from_pred = []
        for output in self.outputs:
            if isinstance(output, UnsupervisedModelOutput):
                continue
            try:
                targets[output.target_property] = batch[output.target_property]
            except KeyError:
                extract_from_pred.append(output.target_property)

            additional_properites = getattr(
                output, "additional_target_properties", None
            )
            if additional_properites is not None:
                for prop in additional_properites:
                    targets[prop] = batch[prop]

        targets[properties.subgraph_mask] = batch.get(properties.subgraph_mask, None)
        try:
            targets[properties.conditions_mask] = batch[properties.conditions_mask]
            targets[properties.conditions_idx_m] = batch[properties.conditions_idx_m]
            targets[properties.conditions_n_atoms] = batch[
                properties.conditions_n_atoms
            ]
        except Exception:
            pass

        # predict output quantity
        pred = self.predict_without_postprocessing(batch)

        # extract missing target values from the prediction
        for prop in extract_from_pred:
            targets[prop] = pred[prop].clone().detach()

        # apply constraints
        pred, targets = self.apply_constraints(pred, targets)

        # calculate the loss
        loss = self.loss_fn(pred, targets)

        # log loss and metrics
        batch_size = batch[properties.idx_m].max().item() + 1
        self.log(
            f"{subset}/loss",
            loss,
            on_step=(subset == "train"),
            on_epoch=(subset != "train"),
            prog_bar=(subset != "train"),
            batch_size=batch_size,
        )
        self.log_metrics(pred, targets, subset, batch_size)
        
        for key in batch:
            if properties.rmsd in key:
                self.log(
                    f"{subset}/{key}",
                    batch[key].mean(),
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                    batch_size=batch_size,
                )

        return loss  # type: ignore

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.FloatTensor]:
        """
        define the training step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # perform forward pass
        loss = self._step(batch, "train")

        # skip exploding batches in backward pass
        if self.skip_exploding_batches and (
            torch.isnan(loss) or torch.isinf(loss) or loss > 1e10
        ):
            msg = (
                f"Loss is {loss} for train batch_idx {batch_idx} and training step "
                f"{self.global_step}, training step will be skipped!"
            )
            if properties.reaction_ids in batch:
                msg += f" Reaction ids: {batch[properties.reaction_ids]}"
            log.warning(msg)
            return None

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.FloatTensor]:
        """
        define the validation step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # enable non-training gradients with respect to specific quanitites if needed.
        torch.set_grad_enabled(self.grad_enabled)

        # forward pass
        loss = self._step(batch, "val")

        return {"val_loss": loss}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.FloatTensor]:
        """
        define the test step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # enable non-training gradients with respect to specific quanitites if needed.
        torch.set_grad_enabled(self.grad_enabled)

        # forward pass
        loss = self._step(batch, "test")

        return {"test_loss": loss}


class ConsiderOnlySelectedAtoms(nn.Module):
    """
    Constraint that allows to neglect some atomic targets (e.g. forces of some specified
    atoms) for model optimization, while not affecting the actual model output. The
    indices of the atoms, which targets to consider in the loss function, must be
    provided in the dataset for each sample in form of a torch tensor of type boolean
    (True: considered, False: neglected).
    """

    def __init__(
        self,
        selection_name: str,
        apply_to_targets: bool = True,
        apply_to_predictions: bool = True,
    ):
        """
        Args:
            selection_name: string associated with the list of considered atoms in the dataset
            apply_to_targets: apply the constraint to the targets.
            apply_to_predictions: apply the constraint to the predictions.
        """
        super().__init__()
        self.selection_name = selection_name
        self.apply_to_targets = apply_to_targets
        self.apply_to_predictions = apply_to_predictions

    def forward(self, pred, targets, output_module):
        """
        A torch tensor is loaded from the dataset, which specifies the considered atoms. Only the
        predictions of those atoms are considered for training, validation, and testing.

        :param pred: python dictionary containing model outputs
        :param targets: python dictionary containing targets
        :param output_module: torch.nn.Module class of a particular property (e.g. forces)
        :return: model outputs and targets of considered atoms only
        """

        if self.selection_name in targets:
            considered_atoms = targets[self.selection_name].nonzero()[:, 0]

            # drop neglected atoms
            if self.apply_to_predictions:
                pred[output_module.name] = pred[output_module.name][considered_atoms]

            if self.apply_to_targets:
                targets[output_module.target_property] = targets[
                    output_module.target_property
                ][considered_atoms]

        return pred, targets
