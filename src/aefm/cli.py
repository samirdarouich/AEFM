# The following script is adapted from SchNetPack's cli.py script.
# https://github.com/atomistic-machine-learning/schnetpack

import logging
import os
import random
import socket
import tempfile
import uuid
from typing import List

import ase
import hydra
import numpy as np
import schnetpack as spk
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning import (
    callbacks as pl_callbacks,
)
from pytorch_lightning.loggers.logger import Logger
from schnetpack.utils import str2class
from schnetpack.utils.script import log_hyperparameters, print_config

from aefm.utils.script import check_existing_config

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)

fields = (
    "run",
    "globals",
    "data",
    "model",
    "task",
    "trainer",
    "callbacks",
    "logger",
    "seed",
    "sampler",
)

header = """
░▒▓███████▓▒░ ▒▓███████▓▒░ ▒▓███████▓▒░ ░▒▓██████████████▓▒░ 
░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒░       ░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ 
░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒░       ░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ 
░▒▓███████▓▒░ ▒▓██████▓▒░  ▒▓██████▓▒░  ░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ 
░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒░       ░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ 
░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒░       ░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ 
░▒▓█▓▒  ▒▓█▓▒ ▒▓████████▓▒ ▒▓█▓▒░       ░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ 
"""


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def train(config: DictConfig):
    """
    General training routine for all models defined by the provided hydra configs.

    """
    print(header)
    log.info("Running on host: " + str(socket.gethostname()))

    if OmegaConf.is_missing(config, "run.data_dir"):
        log.error(
            "Config incomplete! You need to specify the data directory `data_dir`."
        )
        return

    if not ("model" in config and "data" in config):
        log.error(
            """
                Config incomplete! You have to specify at least `data` and `model`!
            """
        )
        return

    # Check for exisiting config files
    check_existing_config(config, "config_train", train=True)

    if config.get("print_config", True):
        print_config(config, fields=fields, resolve=False)

    if "matmul_precision" in config and config.matmul_precision is not None:
        log.info(f"Setting float32 matmul precision to <{config.matmul_precision}>")
        torch.set_float32_matmul_precision(config.matmul_precision)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.info("Seed randomly...")
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
        seed_everything(seed, workers=True)

    if not os.path.exists(config.run.data_dir):
        os.makedirs(config.run.data_dir)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    # Init LightningModule
    log.info(f"Instantiating task <{config.task._target_}>")
    scheduler_cls = (
        str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
    )

    task: spk.AtomisticTask = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )

    # Add pretrained model for fine tuning
    if "pretrained" in config:
        if config.pretrained is not None:
            log.info(f"\n\nLoading pretrained model from <{config.pretrained}>\n\n")
            pretrained = torch.load(config.pretrained, "cpu")

            if isinstance(pretrained, torch.nn.Module):
                state_dict = pretrained.state_dict()
            elif isinstance(pretrained, dict):
                state_dict = pretrained["state_dict"]
                if "model" in list(state_dict.keys())[0]:
                    state_dict = {
                        k.replace("model.", ""): v for k, v in state_dict.items()
                    }
            task.model.load_state_dict(state_dict)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        if config["logger"] is not None:
            for _, lg_conf in config["logger"].items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=os.path.join(config.run.id),
        _convert_="partial",
    )

    log.info("Logging hyperparameters.")
    log_hyperparameters(config=config, model=task, trainer=trainer)

    # Check old ckpt and overwrite early stopping callback
    if config.run.ckpt_path is not None:
        if os.path.exists(config.run.ckpt_path):
            early_stopping = None
            for callback in callbacks:
                if isinstance(callback, pl_callbacks.EarlyStopping):
                    early_stopping = callback
                    break
            if early_stopping is not None:
                log.info(
                    f"Found old checkpoint at <{config.run.ckpt_path}>. Overwriting "
                    "patience of early stopping callback."
                )
                ckpt = torch.load(config.run.ckpt_path, map_location="cpu")
                if "callbacks" in ckpt:
                    for callback, value in ckpt["callbacks"].items():
                        if "EarlyStopping" in callback:
                            value["patience"] = early_stopping.patience
                            break
                    torch.save(ckpt, config.run.ckpt_path)

    # Train the model
    log.info("Starting training.")
    trainer.fit(model=task, datamodule=datamodule, ckpt_path=config.run.ckpt_path)
    
    # Store best model
    log.info("Training completed.")
    best_path = trainer.checkpoint_callback.best_model_path  # type: ignore
    best_score = trainer.checkpoint_callback.best_model_score  # type: ignore
    log.info(f"Best validation score: {best_score.item():.6f}")
    log.info(f"Best checkpoint path:\n<{best_path}>")

    log.info("Store best model")
    best_task = type(task).load_from_checkpoint(best_path)
    torch.save(best_task, config.globals.model_path + ".task")

    best_task.save_model(config.globals.model_path, do_postprocessing=True)
    log.info(f"Best model stored at <{os.path.abspath(config.globals.model_path)}>")
    
    if not config.get("hyperopt", False):
        # Evaluate model on test set after training
        log.info("Starting testing.")
        trainer.test(model=task, datamodule=datamodule, ckpt_path="best")

fields = (
    "run",
    "globals",
    "aefmsampler",
)

@hydra.main(config_path="configs", config_name="sample", version_base="1.2")
def sample(config: DictConfig):
    print(header)
    log.info("Running on host: " + str(socket.gethostname()))

    if OmegaConf.is_missing(config, "globals.model") or OmegaConf.is_missing(
        config, "globals.samples_path"
    ):
        log.error(
            """
                Config incomplete! You have to specify at least `samples_path` and `model`!
            """
        )
        return

    if config.get("print_config", True):
        print_config(config, fields=fields, resolve=False)

    # Init sampler
    log.info(f"Instantiating sampler <{config.aefmsampler._target_}>")
    sampler = hydra.utils.instantiate(config.aefmsampler)

    # Load samples
    log.info(f"Loading samples from <{config.globals.samples_path}>")
    samples = ase.io.read(config.globals.samples_path, index=":")
    log.info(f"Loaded {len(samples)} samples.")

    # Sample
    log.info("Sampling...")
    sampler.sample(samples)
