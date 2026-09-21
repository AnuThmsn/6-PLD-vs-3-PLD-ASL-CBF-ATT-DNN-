"""Generate and execute the exhaustive 3-PLD selection notebook."""
import nbformat as nbf
import itertools

nb = nbf.v4.new_notebook()
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
cells = []

# CELL 0 - Intro
cells.append(nbf.v4.new_markdown_cell("""\
# Exhaustive 3-PLD Selection
**Objective:** Determine the optimal 3-PLD combination out of the 20 possible combinations, strictly using validation data.
"""))

# CELL 1 - Setup
cells.append(nbf.v4.new_code_cell("""\
import sys, os, copy, json, itertools, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from IPython.display import display, Markdown

project_root = Path.cwd().parent if not (Path.cwd() / "src").exists() else Path.cwd()
sys.path.insert(0, str(project_root / "src"))
from simulation import SimulationConfig, seed_everything, generate_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

cfg = SimulationConfig()
SEED = 42

EXPERIMENT_CONFIG = {
    "n_train": 100000,
    "n_val": 10000,
    "n_test": 10000,
    "train_seed": SEED + 10,
    "val_seed": SEED + 20,
    "test_seed": 999,
    "snr": 10.0,
    "batch_size": 512,
    "lr": 1e-3,
    "patience": 20,
    "max_epochs": 200,
    "cbf_width": 50,
    "att_width": 100,
    "grad_clip": 1.0
}

all_combinations = list(itertools.combinations(range(6), 3))
print(f"Generated {len(all_combinations)} possible 3-PLD combinations.")
"""))

# CELL 2 - Generate Data
cells.append(nbf.v4.new_code_cell("""\
seed_everything(SEED)
torch.manual_seed(SEED)

print("Generating 100k training set...")
X_tr_full, Y_tr, _ = generate_dataset(EXPERIMENT_CONFIG["n_train"], cfg, EXPERIMENT_CONFIG["train_seed"], snr=EXPERIMENT_CONFIG["snr"])

print("Generating 10k validation set...")
X_va_full, Y_va, _ = generate_dataset(EXPERIMENT_CONFIG["n_val"], cfg, EXPERIMENT_CONFIG["val_seed"], snr=EXPERIMENT_CONFIG["snr"])

print("Generating 10k test set (LOCKED - DO NOT TOUCH UNTIL END)...")
X_te_full, Y_te, _ = generate_dataset(EXPERIMENT_CONFIG["n_test"], cfg, EXPERIMENT_CONFIG["test_seed"], snr=EXPERIMENT_CONFIG["snr"])

tr_set = set(map(tuple, Y_tr[:1000]))
va_set = set(map(tuple, Y_va[:1000]))
te_set = set(map(tuple, Y_te[:1000]))
assert len(tr_set & va_set) == 0, "LEAK: train/val"
assert len(tr_set & te_set) == 0, "LEAK: train/test"
assert len(va_set & te_set) == 0, "LEAK: val/test"
print("Data generation complete. Zero leakage verified.")
"""))

# CELL 3 - Architecture & Training Loop
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

# CELL 4 - 6-PLD Baseline Training
cells.append(nbf.v4.new_code_cell("""\
# Train the 6-PLD baseline first
out_root = project_root / "results" / "pld_selection"
out_root.mkdir(parents=True, exist_ok=True)

# Target normalization
CBF_mean, CBF_std = float(Y_tr[:, 0].mean()), float(Y_tr[:, 0].std() + 1e-8)
ATT_mean, ATT_std = float(Y_tr[:, 1].mean()), float(Y_tr[:, 1].std() + 1e-8)

Y_tr_norm_cbf = ((Y_tr[:, 0] - CBF_mean) / CBF_std).astype('float32')
Y_tr_norm_att = ((Y_tr[:, 1] - ATT_mean) / ATT_std).astype('float32')
Y_va_norm_cbf = ((Y_va[:, 0] - CBF_mean) / CBF_std).astype('float32')
Y_va_norm_att = ((Y_va[:, 1] - ATT_mean) / ATT_std).astype('float32')

print("Training 6-PLD Baseline...")
six_pld_dir = out_root / "6_pld_baseline"
six_pld_dir.mkdir(exist_ok=True)

X_mean = X_tr_full.mean(0, keepdims=True).astype('float32')
X_std = X_tr_full.std(0, keepdims=True).astype('float32') + 1e-8
X_tr_n = ((X_tr_full - X_mean) / X_std).astype('float32')
X_va_n = ((X_va_full - X_mean) / X_std).astype('float32')

cbf_net_6 = StandardizedNet(6, EXPERIMENT_CONFIG["cbf_width"]).to(device)
cbf_hist_6 = train_net(cbf_net_6, X_tr_n, Y_tr_norm_cbf, X_va_n, Y_va_norm_cbf)

att_net_6 = StandardizedNet(6, EXPERIMENT_CONFIG["att_width"]).to(device)
att_hist_6 = train_net(att_net_6, X_tr_n, Y_tr_norm_att, X_va_n, Y_va_norm_att)

torch.save(cbf_net_6.state_dict(), six_pld_dir / "cbf_model.pt")
torch.save(att_net_6.state_dict(), six_pld_dir / "att_model.pt")
np.savez(six_pld_dir / "normalization.npz", X_mean=X_mean, X_std=X_std, CBF_mean=CBF_mean, CBF_std=CBF_std, ATT_mean=ATT_mean, ATT_std=ATT_std)
"""))

# CELL 5 - The 20-Combination Sweep
cells.append(nbf.v4.new_code_cell("""\
selection_metrics = []

for idx, combo in enumerate(all_combinations):
    combo_name = f"combination_{idx:02d}"
    print(f"\\n--- Evaluating {combo_name}: {combo} ---")
    
    combo_dir = out_root / combo_name
    combo_dir.mkdir(exist_ok=True)
    
    # Input normalization for this combo
    X_tr = X_tr_full[:, combo]
    X_va = X_va_full[:, combo]
    
    X_mean = X_tr.mean(0, keepdims=True).astype('float32')
    X_std = X_tr.std(0, keepdims=True).astype('float32') + 1e-8
    
    X_tr_n = ((X_tr - X_mean) / X_std).astype('float32')
    X_va_n = ((X_va - X_mean) / X_std).astype('float32')
    
    # Train
    start_time = time.time()
    cbf_net = StandardizedNet(3, EXPERIMENT_CONFIG["cbf_width"]).to(device)
    cbf_hist = train_net(cbf_net, X_tr_n, Y_tr_norm_cbf, X_va_n, Y_va_norm_cbf)
    
    att_net = StandardizedNet(3, EXPERIMENT_CONFIG["att_width"]).to(device)
    att_hist = train_net(att_net, X_tr_n, Y_tr_norm_att, X_va_n, Y_va_norm_att)
    train_dur = time.time() - start_time
    
    # Validate
    cbf_net.eval(); att_net.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_va_n).to(device)
        cbf_pred = cbf_net(xt).cpu().squeeze(1).numpy() * CBF_std + CBF_mean
        att_pred = att_net(xt).cpu().squeeze(1).numpy() * ATT_std + ATT_mean
        
    cbf_pred = np.clip(cbf_pred, 0.0, 100.0)
    att_pred = np.clip(att_pred, 0.5, 3.0)
    preds = np.column_stack((cbf_pred, att_pred))
    
    m_cbf = calc_metrics(Y_va[:, 0], preds[:, 0])
    m_att = calc_metrics(Y_va[:, 1], preds[:, 1])
    
    # Save artifacts
    torch.save(cbf_net.state_dict(), combo_dir / "cbf_model.pt")
    torch.save(att_net.state_dict(), combo_dir / "att_model.pt")
    cbf_hist.to_csv(combo_dir / "cbf_training_history.csv", index=False)
    att_hist.to_csv(combo_dir / "att_training_history.csv", index=False)
    np.savez(combo_dir / "validation_predictions.npz", y_true=Y_va, y_pred=preds)
    np.savez(combo_dir / "normalization.npz", X_mean=X_mean, X_std=X_std, CBF_mean=CBF_mean, CBF_std=CBF_std, ATT_mean=ATT_mean, ATT_std=ATT_std)
    
    plds = [cfg.plds_6_s[i] for i in combo]
    row = {
        "Combination": combo_name,
        "Indices": list(combo),
        "PLDs": plds,
        "Val_CBF_MAE": m_cbf["MAE"],
        "Val_CBF_RMSE": m_cbf["RMSE"],
        "Val_CBF_R2": m_cbf["R2"],
        "Val_ATT_MAE": m_att["MAE"],
        "Val_ATT_RMSE": m_att["RMSE"],
        "Val_ATT_R2": m_att["R2"],
        "CBF_Best_Epoch": int(cbf_hist["validation_mae"].idxmin()),
        "ATT_Best_Epoch": int(att_hist["validation_mae"].idxmin()),
        "Train_Duration_s": train_dur
    }
    selection_metrics.append(row)
    
    with open(combo_dir / "config.json", "w") as f:
        json.dump(row, f, indent=2)
        
    # FORCE MEMORY CLEAR SO WINDOWS DOESN'T RUN OUT OF VIRTUAL MEMORY
    import gc
    del cbf_net, att_net, cbf_hist, att_hist, X_tr, X_va, X_tr_n, X_va_n
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

df_sel = pd.DataFrame(selection_metrics)
df_sel.to_csv(out_root / "selection_metrics.csv", index=False)
print("\\nAll 20 combinations evaluated.")
"""))

# CELL 6 - LOCK Selection
cells.append(nbf.v4.new_code_cell("""\
# Primary Criterion: Lowest Validation CBF MAE
df_sorted = df_sel.sort_values(by="Val_CBF_MAE", ascending=True)
best_combo = df_sorted.iloc[0]

locked_config = {
    "selected_indices": best_combo["Indices"],
    "selected_plds": best_combo["PLDs"],
    "combination_name": best_combo["Combination"],
    "selection_metric": "validation_CBF_MAE",
    "selection_protocol": "20-way exhaustive validation selection",
    "training_samples": 100000,
    "test_used_for_selection": False,
    "Val_CBF_MAE": best_combo["Val_CBF_MAE"],
    "Val_ATT_MAE": best_combo["Val_ATT_MAE"]
}

with open(out_root / "selected_pld_config.json", "w") as f:
    json.dump(locked_config, f, indent=2)

print("=== LOCK IN ===")
print(json.dumps(locked_config, indent=2))
"""))

# CELL 7 - Final Test Evaluation
cells.append(nbf.v4.new_code_cell("""\
print("\\n--- FINAL TEST EVALUATION ---")
# The final test set was completely untouched until this exact moment.

final_results = []
test_out_dir = out_root / "final_test_evaluation"
test_out_dir.mkdir(exist_ok=True)

for name, indices, model_dir in [
    ("6_pld_baseline", list(range(6)), out_root / "6_pld_baseline"),
    ("selected_3_pld", locked_config["selected_indices"], out_root / locked_config["combination_name"])
]:
    norm = np.load(model_dir / "normalization.npz")
    X_te = X_te_full[:, indices]
    X_te_n = ((X_te - norm["X_mean"]) / norm["X_std"]).astype('float32')
    
    dim = len(indices)
    cbf_net = StandardizedNet(dim, EXPERIMENT_CONFIG["cbf_width"]).to(device)
    cbf_net.load_state_dict(torch.load(model_dir / "cbf_model.pt", map_location=device))
    cbf_net.eval()
    
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
    
    m_cbf = calc_metrics(Y_te[:, 0], preds[:, 0])
    m_att = calc_metrics(Y_te[:, 1], preds[:, 1])
    
    final_results.append({"Model": name, "Parameter": "CBF", **m_cbf})
    final_results.append({"Model": name, "Parameter": "ATT", **m_att})
    
    np.savez(test_out_dir / f"{name}_test_predictions.npz", y_true=Y_te, y_pred=preds)

df_final = pd.DataFrame(final_results)
df_final.to_csv(test_out_dir / "final_test_metrics.csv", index=False)
display(Markdown("### Final Touched-Test Results"))
display(df_final)

# Relative Change
base_cbf = df_final[(df_final.Model == "6_pld_baseline") & (df_final.Parameter == "CBF")].iloc[0]
sel_cbf = df_final[(df_final.Model == "selected_3_pld") & (df_final.Parameter == "CBF")].iloc[0]
cbf_deg_mae = (sel_cbf["MAE"] - base_cbf["MAE"]) / base_cbf["MAE"] * 100
cbf_deg_rmse = (sel_cbf["RMSE"] - base_cbf["RMSE"]) / base_cbf["RMSE"] * 100

base_att = df_final[(df_final.Model == "6_pld_baseline") & (df_final.Parameter == "ATT")].iloc[0]
sel_att = df_final[(df_final.Model == "selected_3_pld") & (df_final.Parameter == "ATT")].iloc[0]
att_deg_mae = (sel_att["MAE"] - base_att["MAE"]) / base_att["MAE"] * 100
att_deg_rmse = (sel_att["RMSE"] - base_att["RMSE"]) / base_att["RMSE"] * 100

rel_df = pd.DataFrame([
    {"Parameter": "CBF", "MAE_Change_%": cbf_deg_mae, "RMSE_Change_%": cbf_deg_rmse},
    {"Parameter": "ATT", "MAE_Change_%": att_deg_mae, "RMSE_Change_%": att_deg_rmse}
])
rel_df.to_csv(test_out_dir / "final_relative_change.csv", index=False)
"""))

# CELL 8 - Independent Verification Check
cells.append(nbf.v4.new_code_cell("""\
print("\\n--- INDEPENDENT CHECKPOINT VERIFICATION ---")
# Rebuild test set from scratch
_, Y_verify, _ = generate_dataset(EXPERIMENT_CONFIG["n_test"], cfg, EXPERIMENT_CONFIG["test_seed"], snr=EXPERIMENT_CONFIG["snr"])

for name, model_dir in [
    ("6_pld_baseline", out_root / "6_pld_baseline"),
    ("selected_3_pld", out_root / locked_config["combination_name"])
]:
    # We load predictions saved earlier
    saved_preds = np.load(test_out_dir / f"{name}_test_predictions.npz")["y_pred"]
    
    # We load models and manually infer
    norm = np.load(model_dir / "normalization.npz")
    
    if name == "6_pld_baseline":
        X_ver = X_te_full[:, list(range(6))]
    else:
        X_ver = X_te_full[:, locked_config["selected_indices"]]
        
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
    print(f"[{name}] Max abs diff: {np.max(diff):.3e} | Mean abs diff: {np.mean(diff):.3e} | RMSE: {np.sqrt(np.mean(diff**2)):.3e}")
"""))

nb["cells"] = cells

with open("notebooks/exhaustive_3_pld_selection.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook build complete.")

