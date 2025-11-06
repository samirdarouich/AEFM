import optuna
import subprocess
import argparse
import os
import re
import sys
import logging
from functools import partial

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def parse_best_val_score_from_logs(output: str) -> float:
    """
    Extracts the 'Best validation score: <float>' line from logs.
    Returns float('inf') if not found.
    """
    matches = re.findall(r"Best validation score:\s*([0-9.+-eE]+)", output)
    if matches:
        # Convert to float; last one wins if printed multiple times
        return float(matches[-1])
    else:
        return float("inf")
    
def create_run_name(experiment: str, trial_id: int, hyperparams: dict) -> str:
    parts = []
    for key, value in hyperparams.items():
        key_short = key.split('.')[-1]
        if isinstance(value, float):
            if value < 1:
                parts.append(f"{key_short}_{value:.0e}")
            else:
                parts.append(f"{key_short}_{value:.0f}")
        else:
            parts.append(f"{key_short}_{value}")
    log.info(f"Run trial no {trial_id} with parameters: " + ", ".join(parts))
    # wandb restricts the name to be shorter than 128 characters
    return (f"hyperopt_{experiment}_trial_{trial_id}_" + "_".join(parts))[:128]


def objective(trial, experiment: str, max_epochs: int=50, model: str="LEFTNet", data_workdir: str=None, num_workers: int=8) -> float:

    # Suggest hyperparameters
    sigma_fm = trial.suggest_float("sigma_fm", 0.0, 0.2)
    bond_loss_wt = trial.suggest_float("bond_loss_wt", 0.1, 1.0, log=True)
    bond_loss_cutoff = trial.suggest_float("bond_loss_cutoff", 1.5, 3.0)
   
    # Build hyperparameter dictionary
    hyperparams = {
        "globals.sigma_fm": sigma_fm,
        "globals.bond_loss_weight": bond_loss_wt,
        "globals.bond_loss_cutoff": bond_loss_cutoff,
    }
    
    if model == "equiformer":
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
        num_layers = trial.suggest_int("num_layers", 4, 16)
        n_atom_basis = trial.suggest_categorical("n_atom_basis", [64, 128, 256])
        ffn_hidden_channels = trial.suggest_categorical("ffn_hidden_channels", [128, 256, 512])
        lmax = trial.suggest_int("lmax", 2, 6)
        
        hyperparams.update({
            "globals.lr": lr,
            "globals.weight_decay": weight_decay,
            "globals.num_layers": num_layers,
            "globals.n_atom_basis": n_atom_basis,
            "globals.ffn_hidden_channels": ffn_hidden_channels,
            "globals.lmax": lmax,
        })

    
    # Run training via CLI
    run_name = create_run_name(
        experiment=experiment, 
        trial_id=trial.number, 
        hyperparams=hyperparams
    )
    cmd = [
        "aefm_train", 
        f"experiment={experiment}",
        f"run.id={run_name}",
        f"trainer.max_epochs={max_epochs}",
        f"data.num_workers={num_workers}",
        "+trainer.enable_progress_bar=False", # disable progress bar for cleaner logs
    ]

    # Specify the workdir for data if used on a cluster
    if data_workdir is not None:
        cmd.append(f"data.data_workdir={data_workdir}")
    
    # Add hyperparameters to command
    cmd.extend([f"{key}={value}" for key, value in hyperparams.items()])

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if not result.stdout or any("Error" in line for line in result.stderr.split("\n")[-5:]):
        log.warning(f"Something went wrong! Check the setup: {result.stderr}")
        log.warning("Exiting optimization.")
        sys.exit(1)

    # Parse validation score from logs
    val_score = parse_best_val_score_from_logs(result.stdout)

    return val_score

def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization.")
    parser.add_argument("--study_name", type=str, required=True, help="Name of the Optuna study / database file.")
    parser.add_argument("--experiment", type=str, required=True, help="Name of the hydra config to use.")
    parser.add_argument("--model", type=str, default="LEFTNet", help="Model name to optimize. Default: LEFTNet.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials. Default: 20.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs for training. Default: 50.")
    parser.add_argument("--storage_path", type=str, default=None, help="Path to save the optimization results. Default: current directory.")
    parser.add_argument("--data_workdir", type=str, default=None, help="Path to the data working directory.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers. Default: 8.")

    args = parser.parse_args()

    # Configure Optuna logging to stdout
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Build storage path
    storage_path = args.storage_path
    if storage_path is None:
        storage_path = os.getcwd()
    
    # Define storage path and URL
    db_path = f"{storage_path}/hyperopt_{args.study_name}.db"
    storage_url = f"sqlite:///{db_path}"

    # Define objective function with fixed parameters
    objective_fn = partial(
        objective, 
        experiment=args.experiment, 
        max_epochs=args.max_epochs, 
        model=args.model, 
        data_workdir=args.data_workdir, 
        num_workers=args.num_workers
    )

    # Check if the database already exists
    if os.path.exists(db_path):
        log.info(f"Found existing study '{args.study_name}'. Loading from <{db_path}>")
        study = optuna.load_study(study_name=args.study_name, storage=storage_url)
    else:
        log.info(f"Creating new study '{args.study_name}' at <{db_path}>")
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage_url,
            direction="minimize",
            sampler=optuna.samplers.TPESampler()
        )

    # Run optimization
    log.info(f"Starting optimization for model={args.model}, n_trials={args.n_trials}")
    study.optimize(objective_fn, n_trials=args.n_trials)

    # Report best result
    log.info("Optimization completed.")
    log.info("Best hyperparameters:", study.best_params)
    log.info("Best value:", study.best_value)


if __name__ == "__main__":
    main()