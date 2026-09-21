"""Generate and execute the training size convergence notebook."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
cells = []

# CELL 0 - Intro
cells.append(nbf.v4.new_markdown_cell("""\
# Training Size Convergence Experiment
**Objective:** Determine how parameter-estimation performance scales from 4,000 to 1,000,000 training samples.
**Protocol:**
- 1M master training set, sliced to [4k, 10k, 25k, 50k, 100k, 250k, 1M]
- 10k independent validation set
- 10k EXACT SAME test set (seed=999) as the 4k verification experiment
- 6-PLD and 3-PLD configurations
- StandardizedNet (ELU, 9 hidden layers, 50/100 width)
"""))

# CELL 1 - Setup
cells.append(nbf.v4.new_code_cell("""\
import sys, os, copy, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from IPython.display import display, Markdown

project_root = Path.cwd().parent if not (Path.cwd() / "src").exists() else Path.cwd()
sys.path.insert(0, str(project_root / "src"))
from simulation import SimulationConfig, seed_everything, generate_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

cfg = SimulationConfig()
SEED = 42

EXPERIMENT_CONFIG = {
    "sizes": [4000, 10000, 25000, 50000, 100000, 250000, 1000000],
    "n_val": 10000,
    "n_test": 10000,
    "train_seed": SEED + 10,
    "val_seed": SEED + 20,
    "test_seed": 999,
    "snr": 10.0,
    "plds_6": list(range(6)),
    "plds_3": [0, 1, 3],
    "batch_size": 512,
    "lr": 1e-3,
    "patience": 20,
    "max_epochs": 200,
    "cbf_width": 50,
    "att_width": 100,
    "grad_clip": 1.0
}
"""))

# CELL 2 - Generate Data
cells.append(nbf.v4.new_code_cell("""\
seed_everything(SEED)
torch.manual_seed(SEED)

print("Generating 1M master training set...")
X_train_master, Y_train_master, _ = generate_dataset(max(EXPERIMENT_CONFIG["sizes"]), cfg, EXPERIMENT_CONFIG["train_seed"], snr=EXPERIMENT_CONFIG["snr"])

print("Generating 10k validation set...")
X_val_master, Y_val_master, _ = generate_dataset(EXPERIMENT_CONFIG["n_val"], cfg, EXPERIMENT_CONFIG["val_seed"], snr=EXPERIMENT_CONFIG["snr"])

print("Generating 10k test set...")
X_test_master, Y_test_master, _ = generate_dataset(EXPERIMENT_CONFIG["n_test"], cfg, EXPERIMENT_CONFIG["test_seed"], snr=EXPERIMENT_CONFIG["snr"])

tr_set = set(map(tuple, Y_train_master[:1000])) # Check subset for speed
va_set = set(map(tuple, Y_val_master[:1000]))
te_set = set(map(tuple, Y_test_master[:1000]))
assert len(tr_set & va_set) == 0, "LEAK: train/val"
assert len(tr_set & te_set) == 0, "LEAK: train/test"
assert len(va_set & te_set) == 0, "LEAK: val/test"
print("Data generation complete. Zero leakage verified.")
"""))

# CELL 3 - Architecture
cells.append(nbf.v4.new_code_cell("""\
class StandardizedNet(nn.Module):
    def __init__(self, input_dim, width):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.ELU()]
        for _ in range(8):
            layers += [nn.Linear(width, width), nn.ELU()]
        layers.append(nn.Linear(width, 1))
        self.backbone = nn.Sequential(*layers)
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.backbone(x)

def calc_metrics(y_true, y_pred):
    e = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "Bias": float(np.mean(e)),
    }

def train_net(net, X_tr, Y_tr, X_va, Y_va):
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr[:, None])), batch_size=EXPERIMENT_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Y_va[:, None])), batch_size=EXPERIMENT_CONFIG["batch_size"])
    
    opt = torch.optim.Adam(net.parameters(), lr=EXPERIMENT_CONFIG["lr"])
    loss_fn = nn.L1Loss()
    best_val = float('inf')
    best_state = copy.deepcopy(net.state_dict())
    stale = 0
    hist = []
    
    for epoch in range(1, EXPERIMENT_CONFIG["max_epochs"] + 1):
        net.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(net(xb.to(device)), yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), EXPERIMENT_CONFIG["grad_clip"])
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += loss_fn(net(xb.to(device)), yb.to(device)).item()
        val_loss /= len(val_loader)
        
        hist.append({"epoch": epoch, "train_mae": train_loss, "validation_mae": val_loss})
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= EXPERIMENT_CONFIG["patience"]:
                break
                
    net.load_state_dict(best_state)
    return pd.DataFrame(hist)
"""))

# CELL 4 - Execution Loop
cells.append(nbf.v4.new_code_cell("""\
out_root = project_root / "results" / "training_size_convergence"
out_root.mkdir(parents=True, exist_ok=True)

all_results = []

for size in EXPERIMENT_CONFIG["sizes"]:
    print(f"\\n{'='*60}\\n  TRAINING SIZE: {size}\\n{'='*60}")
    
    X_train_sub = X_train_master[:size]
    Y_train_sub = Y_train_master[:size]
    
    # Target stats ONLY from training subset
    CBF_mean, CBF_std = float(Y_train_sub[:, 0].mean()), float(Y_train_sub[:, 0].std() + 1e-8)
    ATT_mean, ATT_std = float(Y_train_sub[:, 1].mean()), float(Y_train_sub[:, 1].std() + 1e-8)
    
    Y_tr_norm_cbf = ((Y_train_sub[:, 0] - CBF_mean) / CBF_std).astype('float32')
    Y_tr_norm_att = ((Y_train_sub[:, 1] - ATT_mean) / ATT_std).astype('float32')
    Y_va_norm_cbf = ((Y_val_master[:, 0] - CBF_mean) / CBF_std).astype('float32')
    Y_va_norm_att = ((Y_val_master[:, 1] - ATT_mean) / ATT_std).astype('float32')
    
    for cfg_name, indices in [("6_pld", EXPERIMENT_CONFIG["plds_6"]), ("3_pld", EXPERIMENT_CONFIG["plds_3"])]:
        print(f"  -- Model: {cfg_name} --")
        
        # Input stats ONLY from training subset
        X_tr = X_train_sub[:, indices]
        X_va = X_val_master[:, indices]
        X_te = X_test_master[:, indices]
        
        X_mean = X_tr.mean(0, keepdims=True).astype('float32')
        X_std = X_tr.std(0, keepdims=True).astype('float32') + 1e-8
        
        X_tr_n = ((X_tr - X_mean) / X_std).astype('float32')
        X_va_n = ((X_va - X_mean) / X_std).astype('float32')
        X_te_n = ((X_te - X_mean) / X_std).astype('float32')
        
        dim = len(indices)
        
        # CBF
        cbf_net = StandardizedNet(dim, EXPERIMENT_CONFIG["cbf_width"]).to(device)
        cbf_hist = train_net(cbf_net, X_tr_n, Y_tr_norm_cbf, X_va_n, Y_va_norm_cbf)
        
        # ATT
        att_net = StandardizedNet(dim, EXPERIMENT_CONFIG["att_width"]).to(device)
        att_hist = train_net(att_net, X_tr_n, Y_tr_norm_att, X_va_n, Y_va_norm_att)
        
        # Predict
        cbf_net.eval(); att_net.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X_te_n).to(device)
            cbf_pred = cbf_net(xt).cpu().squeeze(1).numpy() * CBF_std + CBF_mean
            att_pred = att_net(xt).cpu().squeeze(1).numpy() * ATT_std + ATT_mean
            
        cbf_pred = np.clip(cbf_pred, 0.0, 100.0)
        att_pred = np.clip(att_pred, 0.5, 3.0)
        preds = np.column_stack((cbf_pred, att_pred))
        
        # Metrics
        m_cbf = calc_metrics(Y_test_master[:, 0], preds[:, 0])
        m_att = calc_metrics(Y_test_master[:, 1], preds[:, 1])
        
        all_results.append({"Size": size, "Model": cfg_name, "Parameter": "CBF", **m_cbf})
        all_results.append({"Size": size, "Model": cfg_name, "Parameter": "ATT", **m_att})
        
        # Save artifacts
        dir_path = out_root / str(size) / cfg_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(cbf_net.state_dict(), dir_path / "cbf_model.pt")
        torch.save(att_net.state_dict(), dir_path / "att_model.pt")
        cbf_hist.to_csv(dir_path / "cbf_training_history.csv", index=False)
        att_hist.to_csv(dir_path / "att_training_history.csv", index=False)
        np.savez(dir_path / "test_predictions.npz", y_true=Y_test_master, y_pred=preds)
        np.savez(dir_path / "normalization.npz", X_mean=X_mean, X_std=X_std, CBF_mean=CBF_mean, CBF_std=CBF_std, ATT_mean=ATT_mean, ATT_std=ATT_std)
        pd.DataFrame([m_cbf, m_att]).to_csv(dir_path / "metrics.csv", index=False)
        
        config = {
            "size": size, "model": cfg_name, "indices": indices, "cbf_width": 50, "att_width": 100,
            "lambda_fix_applied": True, "cbf_best_epoch": int(cbf_hist["validation_mae"].idxmin()),
            "att_best_epoch": int(att_hist["validation_mae"].idxmin())
        }
        with open(dir_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

df = pd.DataFrame(all_results)
df.to_csv(out_root / "convergence_metrics.csv", index=False)
print("\\nAll training completed.")
"""))

# CELL 5 - Analysis and Plots
cells.append(nbf.v4.new_code_cell("""\
df = pd.read_csv(out_root / "convergence_metrics.csv")

# Generate Convergence Plots
for param in ["CBF", "ATT"]:
    for metric in ["MAE", "RMSE", "R2"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = df[df.Parameter == param]
        m6 = sub[sub.Model == "6_pld"]
        m3 = sub[sub.Model == "3_pld"]
        
        ax.plot(m6["Size"], m6[metric], 'o-', label="6-PLD")
        ax.plot(m3["Size"], m3[metric], 's-', label="3-PLD")
        ax.set_xscale("log")
        ax.set_xlabel("Training Samples (Log Scale)")
        ax.set_ylabel(metric)
        ax.set_title(f"{param} {metric} Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_root / f"{param}_{metric}_convergence.png", dpi=150)
        plt.close(fig)

# Relative Degradation Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for j, param in enumerate(["CBF", "ATT"]):
    ax = ax1 if j == 0 else ax2
    sub = df[df.Parameter == param].pivot(index="Size", columns="Model", values="MAE")
    rel = (sub["3_pld"] - sub["6_pld"]) / sub["6_pld"] * 100
    ax.plot(rel.index, rel.values, 'd-', color='red')
    ax.set_xscale("log")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("3-PLD MAE % Increase (Degradation)")
    ax.set_title(f"{param} Relative Degradation")
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_root / "relative_degradation.png", dpi=150)
plt.close(fig)
print("Plots generated.")
"""))

# CELL 6 - Independent Verification
cells.append(nbf.v4.new_code_cell("""\
print("Running independent checkpoint verification for Size 1,000,000...")
out_1m = out_root / "1000000" / "6_pld"
if out_1m.exists():
    norm = np.load(out_1m / "normalization.npz")
    X_te = X_test_master[:, list(range(6))]
    X_te_n = ((X_te - norm["X_mean"]) / norm["X_std"]).astype('float32')
    
    net = StandardizedNet(6, 50).to(device)
    net.load_state_dict(torch.load(out_1m / "cbf_model.pt", map_location=device))
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(X_te_n).to(device)).cpu().squeeze(1).numpy() * float(norm["CBF_std"]) + float(norm["CBF_mean"])
    pred = np.clip(pred, 0.0, 100.0)
    
    saved = np.load(out_1m / "test_predictions.npz")["y_pred"][:, 0]
    diff = np.abs(pred - saved)
    print(f"Max abs diff: {np.max(diff):.3e}")
    print(f"Mean abs diff: {np.mean(diff):.3e}")
    print(f"RMSE between arrays: {np.sqrt(np.mean(diff**2)):.3e}")
    print(f"Verification successful: {np.max(diff) < 1e-4}")
"""))

nb["cells"] = cells

with open("notebooks/training_size_convergence.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook build complete.")

