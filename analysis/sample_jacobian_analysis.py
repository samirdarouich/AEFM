import os
import random

import schnetpack
import schnetpack.transform as trn
import torch
import torch.nn as nn
from ase.io import read
from aefm import properties
from aefm.transform import (
    AllToAllNeighborList,
    SubtractCenterOfGeometry,
)
from pytorch_lightning import seed_everything
from schnetpack.interfaces.ase_interface import AtomsConverter

seed_everything(42)
random.seed(42)

def load_model(model):
    model = torch.load(model, device).eval()
    # Check for postprocessors that cast to double precision and remove them
    model.postprocessors = nn.ModuleList(
        [
            postprocessor
            for postprocessor in model.postprocessors
            if not isinstance(postprocessor, schnetpack.transform.CastTo64)
        ]
    )
    try:
        output_key = model.output_modules[0].output_key
    except:
        output_key = output_key
    return model, output_key


def get_jacobian_metrics(f, x_in):
    x = x_in.clone().detach().requires_grad_(True)
    D = x.numel()
    I_D = torch.eye(D, device=x.device)

    y_flat = f(x).view(-1)

    def get_vjp(v):
        return torch.autograd.grad(y_flat, x, v, create_graph=True)[0].view(-1)

    jacobian = torch.vmap(get_vjp)(I_D)
    # Move to CPU before deallocating to free GPU mem
    jacobian_cpu = jacobian.detach().cpu()
    del jacobian, x

    eigvals = torch.linalg.eigvals(jacobian_cpu)

    eigenvalues_mean = eigvals.abs().mean()
    eigenvalues_std = eigvals.abs().std()
    spectral_norm = eigvals.abs().max()
    frob_norm = torch.linalg.norm(jacobian_cpu)

    return (
        spectral_norm,
        eigenvalues_mean,
        eigenvalues_std,
        frob_norm,
    )


def anderson(
    f, x0, m=6, lam=1e-4, threshold=50, eps=1e-3, stop_mode="rel", beta=1.0, **kwargs
):
    """Anderson acceleration for fixed point iteration."""
    bsz, d, L = x0.shape
    alternative_mode = "rel" if stop_mode == "abs" else "abs"
    X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}
    history = [x0]
    for k in range(2, threshold):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        #! alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n) --> deprecated
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[
            :, 1 : n + 1, 0
        ]

        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        history.append(X[:, k % m].reshape_as(x0))
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = (
                        X[:, k % m].view_as(x0).clone().detach(),
                        gx.clone().detach(),
                    )
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    jacobian_metrics = []
    for x_in in history:
        jacobian_metrics.append(get_jacobian_metrics(f, x_in))

    out = {
        "result": lowest_xest,
        "history": torch.stack(history),
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "jacobian_metrics": jacobian_metrics,
        "prot_break": False,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "threshold": threshold,
    }
    X = F = None
    return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Converter for single sample to schnetpack input
converter = AtomsConverter(
    neighbor_list=AllToAllNeighborList(),
    transforms=[
        trn.CastTo64(),
        SubtractCenterOfGeometry(),
        trn.CastTo32(),
    ],
    device=device,
)

eps = 1e-2
solver_kwargs = {
    "threshold": 100,
    "eps": eps,
    "stop_mode": "abs",
    "m": 5,
    "lam": 1e-4,
    "beta": 1.0,
}

## Database

data_source = "xtb_ci_guess"
data_split = "test"
sample_database = read(
    f"/data/{data_source}_"{data_split}.xyz",
    ":",
)

# Select a random subset of x samples
# sample_database = random.sample(sample_database, 100)

# Finetune TS + noise structure to TS
model_type = "sigma"
version = "adaptive_sigma_2_xtb_full_0"
finetune_net = (
    f"training/{model_type}/runs/{version}/best_model"
)

model, output_key = load_model(finetune_net)


print(
    f"Number of samples in {data_source} with split {data_split}: "
    f"{len(sample_database)}"
)

file = "datafiles/jacobian_metrics.pt"
if os.path.exists(file):
    results = torch.load(file, map_location="cpu")
else:
    results = {}

print(f"Already processed {len(results)} reactions.")

for atom in sample_database:
    rxn = atom.info["rxn"]
    
    if rxn == 2422:
        print("Skipping reaction 2422, known problematic case.")
        continue
    
    if rxn in results:
        print(f"Skipping reaction {rxn}, already processed.")
        continue
    print(f"Processing reaction {rxn}...")

    # Convert generated sample back to input and finetune
    inputs = converter(atom)

    # Adapt the convergence based on the number of atoms (relation norm to RMSD)
    if solver_kwargs["stop_mode"] == "abs":
        solver_kwargs["eps"] = eps * torch.sqrt(inputs[properties.n_atoms][0])

    def f(x):
        # x has shape (1, n_atoms, 3)
        inputs[properties.R] = x.squeeze(0)
        # return shape (1, n_atoms, 3)
        return model(inputs)[output_key].unsqueeze(0)

    x0 = inputs[properties.R].unsqueeze(0).clone().detach()

    try:
        out = anderson(
            f,
            x0,
            **solver_kwargs,
        )
        results[rxn] = out["jacobian_metrics"]
    except Exception as e:
        print(f"Error processing reaction {rxn}: {e}")
        print("Stopping.")
        break


torch.save(results, file)
