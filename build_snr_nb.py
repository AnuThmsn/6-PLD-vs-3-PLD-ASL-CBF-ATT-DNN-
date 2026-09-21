"""Generate and execute the selected 3-PLD SNR robustness notebook (Inference Only)."""
import nbformat as nbf
import itertools

nb = nbf.v4.new_notebook()
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
cells = []

# CELL 0 - Intro
cells.append(nbf.v4.new_markdown_cell("""\
# Selected 3-PLD SNR Robustness (Inference Only)
**Objective:** Evaluate the robustness of the *already trained and locked* 6-PLD baseline and the optimal 3-PLD configuration `[0, 2, 3]` across varying noise conditions (SNR 5 to 80).
"""))

# CELL 1 - Setup
cells.append(nbf.v4.new_code_cell("""\
import sys, os, copy, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

EXPERIMENT_CONFIG = {
    "n_test": 10000,
    "test_seed": 999,
    "snrs": [5, 10, 20, 30, 40, 60, 80],
    "plds_6": list(range(6)),
    "plds_3_locked": [0, 2, 3],
    "cbf_width": 50,
    "att_width": 100
}

# Locked Model Paths
model_dir_6 = project_root / "results" / "pld_selection" / "6_pld_baseline"
model_dir_3 = project_root / "results" / "pld_selection" / "combination_04"
"""))

# CELL 2 - Architecture
cells.append(nbf.v4.new_code_cell("""\
class StandardizedNet(nn.Module):
    def __init__(self, input_dim, width):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.ELU()]
        for _ in range(8):
            layers += [nn.Linear(width, width), nn.ELU()]
        layers.append(nn.Linear(width, 1))
        self.backbone = nn.Sequential(*layers)

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
"""))

# CELL 3 - The SNR Inference Sweep
cells.append(nbf.v4.new_code_cell("""\
out_root = project_root / "results" / "snr_robustness"
out_root.mkdir(parents=True, exist_ok=True)

all_results = []

# Load Normalizations (derived entirely from training set)
norm_6 = np.load(model_dir_6 / "normalization.npz")
norm_3 = np.load(model_dir_3 / "normalization.npz")

for snr in EXPERIMENT_CONFIG["snrs"]:
    print(f"\\n{'='*60}\\n  EVALUATING SNR: {snr}\\n{'='*60}")
    
    # 1. Generate EXACT SAME underlying matched data for this specific SNR
    # Crucial: By passing np.full(), we bypass the bug that randomizes the SNR.
    seed_everything(EXPERIMENT_CONFIG["test_seed"])
    snr_array = np.full(EXPERIMENT_CONFIG["n_test"], float(snr), dtype=np.float32)
    X_te_full, Y_te, _ = generate_dataset(EXPERIMENT_CONFIG["n_test"], cfg, EXPERIMENT_CONFIG["test_seed"], snr=snr_array)
    
    snr_dir = out_root / f"snr_{snr}"
    snr_dir.mkdir(exist_ok=True)
    
    for cfg_name, indices, model_dir, norm in [
        ("6_pld", EXPERIMENT_CONFIG["plds_6"], model_dir_6, norm_6), 
        ("3_pld", EXPERIMENT_CONFIG["plds_3_locked"], model_dir_3, norm_3)
    ]:
        print(f"  -- Model: {cfg_name} --")
        
        # Apply the EXACT normalization derived from the 100k training
        X_te = X_te_full[:, indices]
        X_te_n = ((X_te - norm["X_mean"]) / norm["X_std"]).astype('float32')
        
        dim = len(indices)
        
        # Load and Inference CBF
        cbf_net = StandardizedNet(dim, EXPERIMENT_CONFIG["cbf_width"]).to(device)
        cbf_net.load_state_dict(torch.load(model_dir / "cbf_model.pt", map_location=device))
        cbf_net.eval()
        
        # Load and Inference ATT
        att_net = StandardizedNet(dim, EXPERIMENT_CONFIG["att_width"]).to(device)
        att_net.load_state_dict(torch.load(model_dir / "att_model.pt", map_location=device))
        att_net.eval()
        
        with torch.no_grad():
            xt = torch.from_numpy(X_te_n).to(device)
            cbf_pred = cbf_net(xt).cpu().squeeze(1).numpy() * float(norm["CBF_std"]) + float(norm["CBF_mean"])
            att_pred = att_net(xt).cpu().squeeze(1).numpy() * float(norm["ATT_std"]) + float(norm["ATT_mean"])
            
        cbf_pred = np.clip(cbf_pred, 0.0, 100.0)
        att_pred = np.clip(att_pred, 0.5, 3.0)
        preds = np.column_stack((cbf_pred, att_pred))
        
        # Metrics
        m_cbf = calc_metrics(Y_te[:, 0], preds[:, 0])
        m_att = calc_metrics(Y_te[:, 1], preds[:, 1])
        
        all_results.append({"SNR": snr, "Model": cfg_name, "Parameter": "CBF", **m_cbf})
        all_results.append({"SNR": snr, "Model": cfg_name, "Parameter": "ATT", **m_att})
        
        # Save artifacts
        dir_path = snr_dir / cfg_name
        dir_path.mkdir(exist_ok=True)
        
        np.savez(dir_path / "test_predictions.npz", y_true=Y_te, y_pred=preds)
        pd.DataFrame([m_cbf, m_att]).to_csv(dir_path / "metrics.csv", index=False)
        
        config = {
            "snr": snr, "model": cfg_name, "indices": indices, "checkpoint_path": str(model_dir)
        }
        with open(dir_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

df = pd.DataFrame(all_results)
df.to_csv(out_root / "snr_summary.csv", index=False)
print("\\nAll SNR inferences completed.")
"""))

# CELL 4 - Analysis and Plots
cells.append(nbf.v4.new_code_cell("""\
df = pd.read_csv(out_root / "snr_summary.csv")

# Generate Degradation Table
degradations = []
for snr in EXPERIMENT_CONFIG["snrs"]:
    sub = df[df.SNR == snr]
    cbf_6 = sub[(sub.Model == "6_pld") & (sub.Parameter == "CBF")].iloc[0]
    cbf_3 = sub[(sub.Model == "3_pld") & (sub.Parameter == "CBF")].iloc[0]
    att_6 = sub[(sub.Model == "6_pld") & (sub.Parameter == "ATT")].iloc[0]
    att_3 = sub[(sub.Model == "3_pld") & (sub.Parameter == "ATT")].iloc[0]
    
    degradations.append({
        "SNR": snr,
        "CBF_6_MAE": cbf_6["MAE"],
        "CBF_3_MAE": cbf_3["MAE"],
        "CBF_Abs_Diff": cbf_3["MAE"] - cbf_6["MAE"],
        "CBF_MAE_Degradation_%": (cbf_3["MAE"] - cbf_6["MAE"]) / cbf_6["MAE"] * 100,
        "ATT_6_MAE": att_6["MAE"],
        "ATT_3_MAE": att_3["MAE"],
        "ATT_Abs_Diff": att_3["MAE"] - att_6["MAE"],
        "ATT_MAE_Degradation_%": (att_3["MAE"] - att_6["MAE"]) / att_6["MAE"] * 100
    })
df_deg = pd.DataFrame(degradations)
df_deg.to_csv(out_root / "relative_degradations.csv", index=False)

display(Markdown("### Final Results Summary"))
display(df)
display(Markdown("### Degradation Summary"))
display(df_deg)

plot_dir = out_root / "plots"
plot_dir.mkdir(exist_ok=True)

# Generate Plots
for param in ["CBF", "ATT"]:
    for metric in ["MAE", "RMSE", "R2"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = df[df.Parameter == param]
        m6 = sub[sub.Model == "6_pld"]
        m3 = sub[sub.Model == "3_pld"]
        
        ax.plot(m6["SNR"], m6[metric], 'o-', label="6-PLD")
        ax.plot(m3["SNR"], m3[metric], 's-', label="Selected 3-PLD")
        ax.set_xlabel("SNR")
        ax.set_ylabel(metric)
        ax.set_title(f"{param} {metric} vs SNR")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{param}_{metric}_vs_SNR.png", dpi=150)
        plt.close(fig)

# Relative Degradation Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(df_deg["SNR"], df_deg["CBF_MAE_Degradation_%"], 'd-', color='red')
ax1.set_xlabel("SNR")
ax1.set_ylabel("3-PLD CBF MAE % Increase (Degradation)")
ax1.set_title("CBF Relative Degradation vs SNR")
ax1.grid(True, alpha=0.3)

ax2.plot(df_deg["SNR"], df_deg["ATT_MAE_Degradation_%"], 'd-', color='orange')
ax2.set_xlabel("SNR")
ax2.set_ylabel("3-PLD ATT MAE % Increase (Degradation)")
ax2.set_title("ATT Relative Degradation vs SNR")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(plot_dir / "relative_degradation_vs_SNR.png", dpi=150)
plt.close(fig)
print("Plots generated in results/snr_robustness/plots/")
"""))

# CELL 5 - Independent Verification
cells.append(nbf.v4.new_code_cell("""\
print("\\n--- INDEPENDENT CHECKPOINT VERIFICATION ---")
# To prove no data leakage and pure math
verification_snrs = [10, 40, 80]
for snr in verification_snrs:
    print(f"\\nVerifying SNR {snr}...")
    
    # Rebuild test set from scratch
    snr_array = np.full(EXPERIMENT_CONFIG["n_test"], float(snr), dtype=np.float32)
    X_te_full, Y_verify, _ = generate_dataset(EXPERIMENT_CONFIG["n_test"], cfg, EXPERIMENT_CONFIG["test_seed"], snr=snr_array)
    
    for name, indices, model_dir, norm in [
        ("6_pld", EXPERIMENT_CONFIG["plds_6"], model_dir_6, norm_6), 
        ("3_pld", EXPERIMENT_CONFIG["plds_3_locked"], model_dir_3, norm_3)
    ]:
        snr_dir = out_root / f"snr_{snr}" / name
        
        saved_preds = np.load(snr_dir / "test_predictions.npz")["y_pred"]
        
        X_ver = X_te_full[:, indices]
        X_ver_n = ((X_ver - norm["X_mean"]) / norm["X_std"]).astype('float32')
        
        dim = X_ver.shape[1]
        cbf_check = StandardizedNet(dim, EXPERIMENT_CONFIG["cbf_width"]).to(device)
        cbf_check.load_state_dict(torch.load(model_dir / "cbf_model.pt", map_location=device))
        
        att_check = StandardizedNet(dim, EXPERIMENT_CONFIG["att_width"]).to(device)
        att_check.load_state_dict(torch.load(model_dir / "att_model.pt", map_location=device))
        
        with torch.no_grad():
            xt = torch.from_numpy(X_ver_n).to(device)
            cbf_pred = cbf_check(xt).cpu().squeeze(1).numpy() * float(norm["CBF_std"]) + float(norm["CBF_mean"])
            att_pred = att_check(xt).cpu().squeeze(1).numpy() * float(norm["ATT_std"]) + float(norm["ATT_mean"])
        cbf_pred = np.clip(cbf_pred, 0.0, 100.0)
        att_pred = np.clip(att_pred, 0.5, 3.0)
        preds = np.column_stack((cbf_pred, att_pred))
        
        diff = np.abs(preds - saved_preds)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rmse_diff = np.sqrt(np.mean(diff**2))
        print(f"  [{name}] Max diff: {max_diff:.3e} | Mean diff: {mean_diff:.3e} | RMSE diff: {rmse_diff:.3e} | Valid: {max_diff < 1e-4}")
"""))

nb["cells"] = cells

with open("notebooks/selected_3pld_snr_robustness.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook build complete.")
