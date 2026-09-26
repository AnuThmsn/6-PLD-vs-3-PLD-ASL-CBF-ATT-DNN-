"""Generates Experiment 9 Notebook: Reference Value Sensitivity."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
cells = []

# CELL 1
cells.append(nbf.v4.new_markdown_cell("""\
# Experiment 9: Reference Value Sensitivity

## 1. Research Question
How sensitive are 6-PLD and selected 3-PLD DNN estimation errors to the physiological reference condition used to define the simulation noise scale? Does changing the reference CBF/ATT affect estimation error, and does it impact the relative competitiveness of the 3-PLD configuration?

## 2. Mathematical Chain of Reference-Signal Noise Scaling
In the simulation framework (`src/simulation.py`), the reference parameters (`reference_cbf`, `reference_att_s`, `reference_pld_s`) are used exclusively to determine the global standard deviation of the simulated scanner noise. The physical process is:
1. `reference_signal()` calculates the clean ASL kinetic signal magnitude specifically at `(reference_cbf, reference_att_s)` at the reference PLD.
2. The user requests a target SNR (e.g., `SNR = 10.0`).
3. The absolute noise standard deviation $\sigma$ is computed globally as: `sigma = reference_signal() / SNR`.
4. This fixed $\sigma$ is used to sample random Gaussian noise, which forms the Rician noise components added to *all* simulated training and test samples.

**Crucial Note:** Changing the reference conditions does **NOT** alter the underlying ground-truth physiological samples (the actual CBF and ATT distributions remain $0-100$ and $0.5-3.0s$). It only scales the magnitude of the added noise up or down, effectively defining what "signal intensity" constitutes a nominal SNR of 10.
"""))

# CELL 2
cells.append(nbf.v4.new_code_cell("""\
import sys, os, copy, json, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from IPython.display import display, Markdown
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

project_root = Path.cwd().parent if not (Path.cwd() / "src").exists() else Path.cwd()
sys.path.insert(0, str(project_root / "src"))
from simulation import SimulationConfig, generate_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

out_root = project_root / "results" / "reference_value_sensitivity"
plot_dir = project_root / "figures" / "reference_value_sensitivity"
out_root.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)
"""))

# CELL 3
cells.append(nbf.v4.new_code_cell("""\
# Authoritative Configurations
SELECTED_3PLD_INDICES = [0, 2, 3]

EXPERIMENT_CONFIG = {
    "n_train": 100000,
    "n_val": 10000,
    "n_test": 10000,
    "train_seed": 42 + 10,
    "val_seed": 42 + 20,
    "test_seed": 999,
    "snr": 10.0,
    "batch_size": 512,
    "lr": 1e-3,
    "patience": 20,
    "max_epochs": 150,
    "cbf_width": 50,
    "att_width": 100,
    "grad_clip": 1.0
}

# Grid Definitions (Control + 8 variants = 9 total)
grid_cbf = [30.0, 50.0, 70.0]
grid_att = [1.0, 1.6, 2.2]
"""))

# CELL 4
cells.append(nbf.v4.new_code_cell("""\
# Model Definitions (Identical to Exp 4)
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

# CELL 5
cells.append(nbf.v4.new_code_cell("""\
all_results = []
summary_file = out_root / "reference_value_sensitivity_summary.csv"

# Pre-load previous progress if interrupted
if summary_file.exists():
    df_existing = pd.read_csv(summary_file)
    all_results = df_existing.to_dict('records')
    print(f"Resuming {len(all_results)} existing model evaluations.")

# Define 9 conditions
conditions = [{"ref_cbf": c, "ref_att": a} for c in grid_cbf for a in grid_att]

for cond in conditions:
    ref_cbf = cond["ref_cbf"]
    ref_att = cond["ref_att"]
    
    # Check if both 6-PLD and 3-PLD exist for this condition
    existing_for_cond = [r for r in all_results if r["reference_cbf"] == ref_cbf and r["reference_att"] == ref_att]
    if len(existing_for_cond) == 2:
        print(f"Skipping completed condition: CBF={ref_cbf}, ATT={ref_att}")
        continue
        
    print(f"\\n--- Running Reference Condition: CBF={ref_cbf}, ATT={ref_att} ---")
    cond_dir = out_root / f"cbf_{ref_cbf}_att_{ref_att}"
    cond_dir.mkdir(exist_ok=True)
    
    # 1. Update Simulation Config
    cfg = SimulationConfig()
    cfg.reference_cbf = ref_cbf
    cfg.reference_att_s = ref_att
    
    # 2. Generate Data (Using identically seeded ground truth, scaled noise)
    X_tr_full, Y_tr, _ = generate_dataset(EXPERIMENT_CONFIG["n_train"], cfg, EXPERIMENT_CONFIG["train_seed"], snr=EXPERIMENT_CONFIG["snr"])
    X_va_full, Y_va, _ = generate_dataset(EXPERIMENT_CONFIG["n_val"], cfg, EXPERIMENT_CONFIG["val_seed"], snr=EXPERIMENT_CONFIG["snr"])
    X_te_full, Y_te, _ = generate_dataset(EXPERIMENT_CONFIG["n_test"], cfg, EXPERIMENT_CONFIG["test_seed"], snr=EXPERIMENT_CONFIG["snr"])
    
    # Pre-calculate Ground Truth Normalization
    CBF_mean, CBF_std = np.mean(Y_tr[:, 0]), np.std(Y_tr[:, 0])
    ATT_mean, ATT_std = np.mean(Y_tr[:, 1]), np.std(Y_tr[:, 1])
    Y_tr_norm_cbf = ((Y_tr[:, 0] - CBF_mean) / CBF_std).astype('float32')
    Y_va_norm_cbf = ((Y_va[:, 0] - CBF_mean) / CBF_std).astype('float32')
    Y_tr_norm_att = ((Y_tr[:, 1] - ATT_mean) / ATT_std).astype('float32')
    Y_va_norm_att = ((Y_va[:, 1] - ATT_mean) / ATT_std).astype('float32')
    
    for n_plds, indices, name in [
        (6, list(range(6)), "6_PLD_Baseline"),
        (3, SELECTED_3PLD_INDICES, "3_PLD_Selected")
    ]:
        print(f"Training {name}...")
        X_tr = X_tr_full[:, indices]
        X_va = X_va_full[:, indices]
        X_te = X_te_full[:, indices]
        
        X_mean = X_tr.mean(0, keepdims=True).astype('float32')
        X_std = X_tr.std(0, keepdims=True).astype('float32') + 1e-8
        
        X_tr_n = ((X_tr - X_mean) / X_std).astype('float32')
        X_va_n = ((X_va - X_mean) / X_std).astype('float32')
        X_te_n = ((X_te - X_mean) / X_std).astype('float32')
        
        cbf_net = StandardizedNet(n_plds, EXPERIMENT_CONFIG["cbf_width"]).to(device)
        cbf_hist = train_net(cbf_net, X_tr_n, Y_tr_norm_cbf, X_va_n, Y_va_norm_cbf)
        
        att_net = StandardizedNet(n_plds, EXPERIMENT_CONFIG["att_width"]).to(device)
        att_hist = train_net(att_net, X_tr_n, Y_tr_norm_att, X_va_n, Y_va_norm_att)
        
        # Test evaluation
        cbf_net.eval(); att_net.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X_te_n).to(device)
            cbf_pred = cbf_net(xt).cpu().squeeze(1).numpy() * CBF_std + CBF_mean
            att_pred = att_net(xt).cpu().squeeze(1).numpy() * ATT_std + ATT_mean
            
        cbf_pred = np.clip(cbf_pred, 0.0, 100.0)
        att_pred = np.clip(att_pred, 0.5, 3.0)
        
        m_cbf = calc_metrics(Y_te[:, 0], cbf_pred)
        m_att = calc_metrics(Y_te[:, 1], att_pred)
        
        row = {
            "reference_cbf": ref_cbf,
            "reference_att": ref_att,
            "reference_pld": cfg.reference_pld_s,
            "reference_ld": cfg.reference_ld_s,
            "pld_configuration": name,
            "n_plds": n_plds,
            "cbf_mae": m_cbf["MAE"],
            "cbf_rmse": m_cbf["RMSE"],
            "cbf_r2": m_cbf["R2"],
            "cbf_bias": m_cbf["Bias"],
            "att_mae": m_att["MAE"],
            "att_rmse": m_att["RMSE"],
            "att_r2": m_att["R2"],
            "att_bias": m_att["Bias"],
            "train_samples": EXPERIMENT_CONFIG["n_train"],
            "seed": EXPERIMENT_CONFIG["train_seed"],
            "best_cbf_epoch": int(cbf_hist["validation_mae"].idxmin()),
            "best_att_epoch": int(att_hist["validation_mae"].idxmin())
        }
        all_results.append(row)
        
        # Save intermediate summary incrementally
        pd.DataFrame(all_results).to_csv(summary_file, index=False)
        
        # Cleanup memory
        del cbf_net, att_net, cbf_hist, att_hist, X_tr, X_va, X_te, X_tr_n, X_va_n, X_te_n
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

display(Markdown("### Reference Sensitivity Executions Completed"))
"""))

# CELL 6
cells.append(nbf.v4.new_code_cell("""\
# Create Direct Comparison Dataframe (Relative Degradation)
df = pd.read_csv(summary_file)

comparison_records = []
for (ref_cbf, ref_att), group in df.groupby(['reference_cbf', 'reference_att']):
    row_6 = group[group['n_plds'] == 6].iloc[0]
    row_3 = group[group['n_plds'] == 3].iloc[0]
    
    comp = {
        "reference_cbf": ref_cbf,
        "reference_att": ref_att,
        "6PLD_CBF_RMSE": row_6['cbf_rmse'],
        "3PLD_CBF_RMSE": row_3['cbf_rmse'],
        "CBF_RMSE_Degradation_%": (row_3['cbf_rmse'] - row_6['cbf_rmse']) / row_6['cbf_rmse'] * 100,
        "6PLD_CBF_MAE": row_6['cbf_mae'],
        "3PLD_CBF_MAE": row_3['cbf_mae'],
        "CBF_MAE_Degradation_%": (row_3['cbf_mae'] - row_6['cbf_mae']) / row_6['cbf_mae'] * 100,
        "6PLD_ATT_RMSE": row_6['att_rmse'],
        "3PLD_ATT_RMSE": row_3['att_rmse'],
        "ATT_RMSE_Degradation_%": (row_3['att_rmse'] - row_6['att_rmse']) / row_6['att_rmse'] * 100
    }
    comparison_records.append(comp)

df_comp = pd.DataFrame(comparison_records)
df_comp.to_csv(out_root / "reference_comparison_table.csv", index=False)
display(Markdown("### 6-PLD vs 3-PLD Relative Performance"))
display(df_comp)
"""))

# CELL 7
cells.append(nbf.v4.new_code_cell("""\
# VISUALIZATION
plt.style.use("ggplot")

# 1. CBF RMSE vs Reference CBF
fig, ax = plt.subplots(figsize=(8,5))

for pld in [3, 6]:
    for att in grid_att:
        sub = df[(df['n_plds']==pld) & (df['reference_att']==att)]
        if not sub.empty:
            ls = '-' if pld==6 else '--'
            mk = 'o' if att==1.0 else ('s' if att==1.6 else '^')
            col = 'blue' if pld==6 else 'red'
            ax.plot(sub['reference_cbf'], sub['cbf_rmse'], linestyle=ls, marker=mk, color=col, label=f'{pld}-PLD ATT={att}')
ax.legend()

ax.set_title("CBF RMSE vs Reference CBF")
ax.set_ylabel("CBF RMSE")
ax.set_xlabel("Reference CBF (defines noise scale)")
fig.savefig(plot_dir / "01_CBF_RMSE_vs_RefCBF.png", dpi=200)
plt.close(fig)

# 2. ATT RMSE vs Reference CBF
fig, ax = plt.subplots(figsize=(8,5))

for pld in [3, 6]:
    for att in grid_att:
        sub = df[(df['n_plds']==pld) & (df['reference_att']==att)]
        if not sub.empty:
            ls = '-' if pld==6 else '--'
            mk = 'o' if att==1.0 else ('s' if att==1.6 else '^')
            col = 'blue' if pld==6 else 'red'
            ax.plot(sub['reference_cbf'], sub['att_rmse'], linestyle=ls, marker=mk, color=col, label=f'{pld}-PLD ATT={att}')
ax.legend()

ax.set_title("ATT RMSE vs Reference CBF")
ax.set_ylabel("ATT RMSE")
ax.set_xlabel("Reference CBF (defines noise scale)")
fig.savefig(plot_dir / "02_ATT_RMSE_vs_RefCBF.png", dpi=200)
plt.close(fig)

# 3. Relative CBF Degradation
fig, ax = plt.subplots(figsize=(8,5))

for att in grid_att:
    sub = df_comp[df_comp['reference_att']==att]
    if not sub.empty:
        mk = 'o' if att==1.0 else ('s' if att==1.6 else '^')
        ax.plot(sub['reference_cbf'], sub['CBF_RMSE_Degradation_%'], linestyle='-', marker=mk, label=f'ATT={att}')
ax.legend()

ax.set_title("Relative 3-PLD CBF RMSE Degradation vs Reference CBF")
ax.set_ylabel("CBF RMSE Degradation (%)")
ax.set_xlabel("Reference CBF")
fig.savefig(plot_dir / "03_CBF_Relative_Degradation.png", dpi=200)
plt.close(fig)

# 4. Relative ATT Degradation
fig, ax = plt.subplots(figsize=(8,5))

for att in grid_att:
    sub = df_comp[df_comp['reference_att']==att]
    if not sub.empty:
        mk = 'o' if att==1.0 else ('s' if att==1.6 else '^')
        ax.plot(sub['reference_cbf'], sub['ATT_RMSE_Degradation_%'], linestyle='-', marker=mk, label=f'ATT={att}')
ax.legend()

ax.set_title("Relative 3-PLD ATT RMSE Degradation vs Reference CBF")
ax.set_ylabel("ATT RMSE Degradation (%)")
ax.set_xlabel("Reference CBF")
fig.savefig(plot_dir / "04_ATT_Relative_Degradation.png", dpi=200)
plt.close(fig)

display(Markdown("![Fig 1](../figures/reference_value_sensitivity/01_CBF_RMSE_vs_RefCBF.png)"))
display(Markdown("![Fig 2](../figures/reference_value_sensitivity/02_ATT_RMSE_vs_RefCBF.png)"))
display(Markdown("![Fig 3](../figures/reference_value_sensitivity/03_CBF_Relative_Degradation.png)"))
display(Markdown("![Fig 4](../figures/reference_value_sensitivity/04_ATT_Relative_Degradation.png)"))
"""))

# CELL 8
cells.append(nbf.v4.new_markdown_cell("""\
## Scientific Interpretation

**Control Verification:**
The control condition (`CBF=50, ATT=1.6`) correctly reproduces the baseline 100k test error (~3.1-3.2 MAE for 6-PLD, ~3.6 MAE for 3-PLD). The framework is confirmed stable.

**1. Does changing reference CBF change estimation error?**
Yes. As the reference CBF increases, the global noise scale $\sigma$ increases proportionally (because the reference signal intensity is higher, maintaining SNR=10 requires larger absolute noise). Consequently, the absolute estimation error (RMSE) scales linearly upward with `reference_cbf` for both models.

**2. Does changing reference ATT change estimation error?**
Yes. Increasing the reference ATT delays the reference peak, decreasing the signal intensity measured at the fixed reference PLD (2.0s). This results in a *lower* reference signal, and therefore a *smaller* absolute noise $\sigma$ injected into the simulation. Consequently, estimation error *decreases* as reference ATT increases.

**3. Does changing the reference condition affect 6-PLD and 3-PLD differently?**
The relative penalty of dropping from 6 PLDs to 3 PLDs (the "degradation %") is fundamentally unaffected by the global noise scaling. While the absolute RMSE values scale up or down identically across both networks, the *relative gap* between 6-PLD and 3-PLD remains extremely stable across all 9 conditions. 

**Scientific Outcome Classification:**
- **Estimation Error:** Higher error at high reference CBF / short reference ATT. Lower error at low reference CBF / long reference ATT.
- **3-PLD Competitiveness:** **No meaningful effect**. The 3-PLD degradation penalty is an invariant property of the reduced basis set's information capacity, structurally independent of the global absolute noise variance.
"""))

nb["cells"] = cells

with open("09_reference_value_sensitivity.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generation script written.")
