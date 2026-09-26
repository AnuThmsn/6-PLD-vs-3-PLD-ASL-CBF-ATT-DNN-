"""Generates Experiment 12 Notebook: Signal Identifiability and Reconstruction Audit."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
cells = []

# CELL 1: Markdown Setup
cells.append(nbf.v4.new_markdown_cell("""\
# Experiment 12: Signal Identifiability and Inverse Problem Audit
"""))

# CELL 2: Imports and Data Prep
cells.append(nbf.v4.new_code_cell("""\
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from IPython.display import display, Markdown
import torch
import torch.nn as nn

project_root = Path.cwd().parent if not (Path.cwd() / "src").exists() else Path.cwd()
sys.path.insert(0, str(project_root / "src"))
from simulation import SimulationConfig, generate_dataset, paper_signal

out_root = project_root / "results" / "identifiability_audit"
plot_dir = project_root / "figures" / "identifiability_audit"
out_root.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

cfg = SimulationConfig(reference_cbf=50.0, reference_att_s=1.6)
X_te_full, Y_te, _ = generate_dataset(10000, cfg, seed=999, snr=10.0)

clean_te_full = paper_signal(Y_te[:,0], Y_te[:,1], cfg.plds_6_s, cfg)

# FIX: pass plds as list [2.0] and extract scalar
reference_val = paper_signal(np.array([cfg.reference_cbf]), np.array([cfg.reference_att_s]), [2.0], cfg)[0, 0]
noise_sigma = reference_val / 10.0
display(Markdown(f"**Global Noise Standard Deviation ($\\\\sigma$):** {noise_sigma:.3f}"))
"""))

# CELL 3: Signal Sensitivity Analysis (Jacobian)
cells.append(nbf.v4.new_code_cell("""\
# Create a grid of parameters
cbf_range = np.linspace(20, 90, 50)
att_range = np.linspace(0.5, 3.0, 50)
C, A = np.meshgrid(cbf_range, att_range)

eps_cbf = 1.0 # 1 ml/100g/min change
eps_att = 0.1 # 100 ms change

sensitivity_cbf = np.zeros((50, 50))
sensitivity_att = np.zeros((50, 50))

plds = np.array(cfg.plds_6_s)

for i in range(50):
    for j in range(50):
        c = np.array([C[i, j]])
        a = np.array([A[i, j]])
        
        base_sig = paper_signal(c, a, plds, cfg)[0]
        cbf_sig = paper_signal(c + eps_cbf, a, plds, cfg)[0]
        att_sig = paper_signal(c, a + eps_att, plds, cfg)[0]
        
        sensitivity_cbf[i, j] = np.sqrt(np.sum((cbf_sig - base_sig)**2)) / eps_cbf
        sensitivity_att[i, j] = np.sqrt(np.sum((att_sig - base_sig)**2)) / eps_att

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

c1 = ax1.contourf(C, A, sensitivity_cbf, levels=20, cmap='viridis')
fig.colorbar(c1, ax=ax1)
ax1.set_title(f"Signal Sensitivity to CBF (||ΔS|| per 1 ml/100g/min)")

c2 = ax2.contourf(C, A, sensitivity_att, levels=20, cmap='plasma')
fig.colorbar(c2, ax=ax2)
ax2.set_title(f"Signal Sensitivity to ATT (||ΔS|| per 0.1s)")

plt.tight_layout()
fig.savefig(plot_dir / "01_signal_sensitivity.png", dpi=200)
plt.close(fig)
"""))

# CELL 4: Signal Reconstruction Test
cells.append(nbf.v4.new_code_cell("""\
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_tr, Y_tr, _ = generate_dataset(100000, cfg, seed=42, snr=10.0)
CBF_m, CBF_s = Y_tr[:,0].mean(), Y_tr[:,0].std()
ATT_m, ATT_s = Y_tr[:,1].mean(), Y_tr[:,1].std()

Y_tr_norm = np.column_stack(((Y_tr[:,0]-CBF_m)/CBF_s, (Y_tr[:,1]-ATT_m)/ATT_s))
X_m = X_tr.mean(0, keepdims=True)
X_s = X_tr.std(0, keepdims=True) + 1e-8
X_tr_n = (X_tr - X_m) / X_s
X_te_n = (X_te_full - X_m) / X_s

class BaselineMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        layers = [nn.Linear(in_dim, 100), nn.ELU()]
        for _ in range(8): layers += [nn.Linear(100, 100), nn.ELU()]
        self.net = nn.Sequential(*layers, nn.Linear(100, out_dim))
    def forward(self, x): return self.net(x)

net_cbf = BaselineMLP(6, 1).to(device)
net_att = BaselineMLP(6, 1).to(device)
opt_cbf = torch.optim.Adam(net_cbf.parameters(), lr=1e-3)
opt_att = torch.optim.Adam(net_att.parameters(), lr=1e-3)

xb = torch.from_numpy(X_tr_n).float().to(device)
yb_c = torch.from_numpy(Y_tr_norm[:,0:1]).float().to(device)
yb_a = torch.from_numpy(Y_tr_norm[:,1:2]).float().to(device)

train_loader = DataLoader(TensorDataset(xb, yb_c, yb_a), batch_size=1024, shuffle=True)
for epoch in range(15):
    for batch_x, batch_yc, batch_ya in train_loader:
        opt_cbf.zero_grad()
        nn.L1Loss()(net_cbf(batch_x), batch_yc).backward()
        opt_cbf.step()
        opt_att.zero_grad()
        nn.L1Loss()(net_att(batch_x), batch_ya).backward()
        opt_att.step()

net_cbf.eval(); net_att.eval()
with torch.no_grad():
    preds_c = net_cbf(torch.from_numpy(X_te_n).float().to(device)).cpu().numpy().flatten()
    preds_a = net_att(torch.from_numpy(X_te_n).float().to(device)).cpu().numpy().flatten()

pred_cbf = np.clip(preds_c * CBF_s + CBF_m, 0.0, 100.0)
pred_att = np.clip(preds_a * ATT_s + ATT_m, 0.5, 3.0)

cbf_rmse = np.sqrt(np.mean((pred_cbf - Y_te[:, 0])**2))
att_rmse = np.sqrt(np.mean((pred_att - Y_te[:, 1])**2))

pred_signals = paper_signal(pred_cbf, pred_att, cfg.plds_6_s, cfg)
sig_rmse_vs_clean = np.sqrt(np.mean((pred_signals - clean_te_full)**2))
sig_rmse_vs_noisy = np.sqrt(np.mean((pred_signals - X_te_full)**2))

print(f"CBF RMSE: {cbf_rmse:.3f}, ATT RMSE: {att_rmse:.3f}")
print(f"Signal RMSE (vs Clean): {sig_rmse_vs_clean:.3f} (Noise Level Sigma: {noise_sigma:.3f})")
print(f"Signal RMSE (vs Noisy): {sig_rmse_vs_noisy:.3f}")
"""))

# CELL 5: Iso-Signal Contours
cells.append(nbf.v4.new_code_cell("""\
target_sig = paper_signal(np.array([50.0]), np.array([1.6]), cfg.plds_6_s, cfg)[0]
signal_distance = np.zeros((50, 50))

for i in range(50):
    for j in range(50):
        test_sig = paper_signal(np.array([C[i, j]]), np.array([A[i, j]]), cfg.plds_6_s, cfg)[0]
        signal_distance[i, j] = np.sqrt(np.mean((test_sig - target_sig)**2))

fig, ax = plt.subplots(figsize=(8, 6))
c = ax.contourf(C, A, signal_distance, levels=np.linspace(0, noise_sigma*2, 20), cmap='viridis_r', extend='max')
ax.contour(C, A, signal_distance, levels=[noise_sigma], colors='red', linewidths=2, linestyles='dashed')
fig.colorbar(c, ax=ax, label="Signal RMSE vs Target")
ax.scatter([50], [1.6], color='white', marker='*', s=200, edgecolor='black', label="Target")
ax.set_title(f"Iso-Signal Region (Red Line = Noise Floor $\\\\sigma={noise_sigma:.1f}$)")
plt.tight_layout()
fig.savefig(plot_dir / "02_iso_signal_valley.png", dpi=200)
plt.close(fig)
"""))

nb["cells"] = cells
with open("12_identifiability_and_signal_reconstruction_audit.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook 12 generator written.")
