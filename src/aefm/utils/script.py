import logging
import os

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def check_existing_config(config: DictConfig, config_name: str, train: bool = False):
    config_file = f"{config_name}.yaml"
    if os.path.exists(config_file):
        if "train" in config_name:
            log.info(
                f"Config already exists in given directory {os.path.abspath('.')}."
                + " Attempting to continue training."
            )
        # save old config
        old_config = OmegaConf.load(config_file)
        count = 1
        while os.path.exists(f"{config_name}.old.{count}.yaml"):
            count += 1
        with open(f"{config_name}.old.{count}.yaml", "w") as f:
            OmegaConf.save(old_config, f, resolve=False)

        if train:
            # resume from latest checkpoint
            if config.run.ckpt_path is None:
                if os.path.exists("checkpoints/last.ckpt"):
                    config.run.ckpt_path = "checkpoints/last.ckpt"

            if config.run.ckpt_path is not None:
                log.info(
                    f"Resuming from checkpoint {os.path.abspath(config.run.ckpt_path)}"
                )
    else:
        with open(config_file, "w") as f:
            OmegaConf.save(config, f, resolve=False)
