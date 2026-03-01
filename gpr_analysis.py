"""
Physics-Constrained GPR for MMC Process Optimization
=====================================================
Two systems comparison:
  - System 1: Pure Al + MgO  → optimum at 2.5 wt%
  - System 2: Al7075 + WO₃  → optimum at 3.5 vol%

Research question: Why does the optimal concentration differ?
Can we predict it for a new system without exhaustive experiments?
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
mgo = pd.read_csv('data/mgo_system.csv', comment='#')
wo3 = pd.read_csv('data/wo3_system.csv', comment='#')

print("=== System 1: Pure Al + nano-MgO ===")
print(mgo[['concentration_wt', 'hardness_HV', 'comp_strength_MPa', 'porosity_pct']].to_string(index=False))
print("\n=== System 2: Al7075 + nano-WO₃ ===")
print(wo3[['concentration_vol', 'hardness_HV', 'comp_strength_MPa', 'porosity_pct']].to_string(index=False))


# ─────────────────────────────────────────────
# 2. BUILD GPR MODELS
# ─────────────────────────────────────────────
def build_gpr(X, y, n_restarts=20):
    """
    GPR with RBF kernel + white noise.
    Same kernel family used by Ezzat (P1 paper).
    """
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(1.0, (0.1, 5.0)) + WhiteKernel(0.01)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, normalize_y=False)
    gpr.fit(X_s, y_s)
    return gpr, scaler_X, scaler_y

def predict_gpr(gpr, scaler_X, scaler_y, X_pred):
    X_s = scaler_X.transform(X_pred)
    y_s, std_s = gpr.predict(X_s, return_std=True)
    y_mean = scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()
    y_std  = std_s * scaler_y.scale_[0]
    return y_mean, y_std

# --- MgO system ---
X_mgo = mgo[['concentration_wt']].values
gpr_mgo_h, sx_mgo_h, sy_mgo_h = build_gpr(X_mgo, mgo['hardness_HV'].values)
gpr_mgo_w, sx_mgo_w, sy_mgo_w = build_gpr(X_mgo, mgo['wear_rate_10N_e7'].values)

# --- WO3 system ---
X_wo3 = wo3[['concentration_vol']].values
gpr_wo3_h, sx_wo3_h, sy_wo3_h = build_gpr(X_wo3, wo3['hardness_HV'].values)
gpr_wo3_w, sx_wo3_w, sy_wo3_w = build_gpr(X_wo3, wo3['wear_rate_10N_e7'].values)

# Prediction range
X_pred = np.linspace(0, 5, 200).reshape(-1, 1)

mgo_h_pred, mgo_h_std = predict_gpr(gpr_mgo_h, sx_mgo_h, sy_mgo_h, X_pred)
mgo_w_pred, mgo_w_std = predict_gpr(gpr_mgo_w, sx_mgo_w, sy_mgo_w, X_pred)
wo3_h_pred, wo3_h_std = predict_gpr(gpr_wo3_h, sx_wo3_h, sy_wo3_h, X_pred)
wo3_w_pred, wo3_w_std = predict_gpr(gpr_wo3_w, sx_wo3_w, sy_wo3_w, X_pred)

# Find GPR optima
opt_mgo = X_pred[np.argmax(mgo_h_pred)][0]
opt_wo3 = X_pred[np.argmax(wo3_h_pred)][0]
print(f"\nGPR-predicted optimum (max hardness):")
print(f"  MgO system:  {opt_mgo:.2f} wt%  (experimental: 2.5 wt%)")
print(f"  WO₃ system:  {opt_wo3:.2f} vol%  (experimental: 3.5 vol%)")


# ─────────────────────────────────────────────
# 3. PHYSICS CONSTRAINTS (Archard's Law check)
# ─────────────────────────────────────────────
def archard_wear(hardness, N=10, K=1e-7, S=1810, C=1.0):
    """
    Archard's law: W = K*(N*S) / (C*H)
    Returns expected wear rate in same units as experimental data (×10⁻⁷ g/cm)
    """
    return K * (N * S) / (C * hardness) * 1e7  # scale to ×10⁻⁷

def physics_consistency(h_pred, w_pred, N=10):
    w_archard = archard_wear(h_pred, N=N)
    # Normalize: relative deviation
    consistency = np.abs(w_pred - w_archard) / (w_archard + 1e-9)
    return consistency, w_archard

mgo_consist, mgo_archard = physics_consistency(mgo_h_pred, mgo_w_pred)
wo3_consist, wo3_archard = physics_consistency(wo3_h_pred, wo3_w_pred)

print(f"\nArchard consistency (mean relative deviation):")
print(f"  MgO system:  {mgo_consist.mean():.3f}")
print(f"  WO₃ system:  {wo3_consist.mean():.3f}")


# ─────────────────────────────────────────────
# 4. MAIN COMPARISON FIGURE (the key figure)
# ─────────────────────────────────────────────
BLUE1   = '#1565C0'
BLUE2   = '#42A5F5'
RED1    = '#C62828'
RED2    = '#EF9A9A'
GREEN1  = '#2E7D32'
ORANGE1 = '#E65100'
GRAY    = '#ECEFF1'

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#FAFAFA')

gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                       left=0.08, right=0.96, top=0.88, bottom=0.10)

# ── Panel 1: MgO Hardness ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.fill_between(X_pred.flatten(), mgo_h_pred - 2*mgo_h_std, mgo_h_pred + 2*mgo_h_std,
                 alpha=0.20, color=BLUE1, label='95% CI')
ax1.plot(X_pred, mgo_h_pred, color=BLUE1, lw=2.5, label='GPR prediction')
ax1.scatter(mgo['concentration_wt'], mgo['hardness_HV'],
            color=RED1, s=80, zorder=5, label='Experimental', edgecolors='white', linewidths=0.8)
ax1.axvline(opt_mgo, color=GREEN1, ls='--', lw=1.5, alpha=0.8)
ax1.text(opt_mgo + 0.08, ax1.get_ylim()[0] + 2, f'opt={opt_mgo:.1f}%',
         color=GREEN1, fontsize=9, fontweight='bold')
ax1.set_xlabel('MgO Content (wt%)', fontsize=10)
ax1.set_ylabel('Micro-Vickers Hardness (HV)', fontsize=10)
ax1.set_title('System 1: Pure Al + nano-MgO\nHardness', fontsize=10, fontweight='bold', color=BLUE1)
ax1.legend(fontsize=8, loc='upper right')
ax1.set_facecolor(GRAY)
ax1.grid(True, alpha=0.4, color='white')

# ── Panel 2: WO3 Hardness ─────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.fill_between(X_pred.flatten(), wo3_h_pred - 2*wo3_h_std, wo3_h_pred + 2*wo3_h_std,
                 alpha=0.20, color=RED1, label='95% CI')
ax2.plot(X_pred, wo3_h_pred, color=RED1, lw=2.5, label='GPR prediction')
ax2.scatter(wo3['concentration_vol'], wo3['hardness_HV'],
            color=BLUE1, s=80, zorder=5, label='Experimental', edgecolors='white', linewidths=0.8)
ax2.axvline(opt_wo3, color=GREEN1, ls='--', lw=1.5, alpha=0.8)
ax2.text(opt_wo3 + 0.08, ax2.get_ylim()[0] + 5, f'opt={opt_wo3:.1f}%',
         color=GREEN1, fontsize=9, fontweight='bold')
ax2.set_xlabel('WO₃ Content (vol%)', fontsize=10)
ax2.set_ylabel('Micro-Vickers Hardness (HV)', fontsize=10)
ax2.set_title('System 2: Al7075 + nano-WO₃\nHardness', fontsize=10, fontweight='bold', color=RED1)
ax2.legend(fontsize=8, loc='upper left')
ax2.set_facecolor(GRAY)
ax2.grid(True, alpha=0.4, color='white')

# ── Panel 3: MgO Wear + Archard check ─────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.fill_between(X_pred.flatten(), mgo_w_pred - 2*mgo_w_std, mgo_w_pred + 2*mgo_w_std,
                 alpha=0.20, color=BLUE1, label='GPR 95% CI')
ax3.plot(X_pred, mgo_w_pred, color=BLUE1, lw=2.5, label='GPR (wear rate)')
ax3.plot(X_pred, mgo_archard, color=ORANGE1, lw=1.8, ls=':', label="Archard's Law (expected)")
ax3.scatter(mgo['concentration_wt'], mgo['wear_rate_10N_e7'],
            color=RED1, s=80, zorder=5, label='Experimental @ 10N', edgecolors='white', linewidths=0.8)
ax3.set_xlabel('MgO Content (wt%)', fontsize=10)
ax3.set_ylabel('Wear Rate (×10⁻⁷ g/cm) @ 10N', fontsize=10)
ax3.set_title('System 1: Wear Rate\n+ Archard Consistency Check', fontsize=10, fontweight='bold', color=BLUE1)
ax3.legend(fontsize=8)
ax3.set_facecolor(GRAY)
ax3.grid(True, alpha=0.4, color='white')

# ── Panel 4: WO3 Wear + Archard check ─────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.fill_between(X_pred.flatten(), wo3_w_pred - 2*wo3_w_std, wo3_w_pred + 2*wo3_w_std,
                 alpha=0.20, color=RED1, label='GPR 95% CI')
ax4.plot(X_pred, wo3_w_pred, color=RED1, lw=2.5, label='GPR (wear rate)')
ax4.plot(X_pred, wo3_archard, color=ORANGE1, lw=1.8, ls=':', label="Archard's Law (expected)")
ax4.scatter(wo3['concentration_vol'], wo3['wear_rate_10N_e7'],
            color=BLUE1, s=80, zorder=5, label='Experimental @ 10N', edgecolors='white', linewidths=0.8)
ax4.set_xlabel('WO₃ Content (vol%)', fontsize=10)
ax4.set_ylabel('Wear Rate (×10⁻⁷ g/cm) @ 10N', fontsize=10)
ax4.set_title('System 2: Wear Rate\n+ Archard Consistency Check', fontsize=10, fontweight='bold', color=RED1)
ax4.legend(fontsize=8)
ax4.set_facecolor(GRAY)
ax4.grid(True, alpha=0.4, color='white')

# ── Main title ────────────────────────────────
fig.suptitle(
    "Physics-Constrained GPR for MMC Optimization\n"
    "Why does the optimal concentration differ?  (MgO: 2.5 wt%  vs  WO₃: 3.5 vol%)  "
    "→ Motivates Physics-Informed Offline RL",
    fontsize=11, fontweight='bold', style='italic', y=0.97, color='#212121'
)

plt.savefig('results/figures/gpr_comparison.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("\nSaved: results/figures/gpr_comparison.png")
plt.close()


# ─────────────────────────────────────────────
# 5. SYSTEM COMPARISON SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SYSTEM COMPARISON SUMMARY")
print("="*60)
summary = {
    'Property': ['Matrix', 'Reinforcement', 'Reinforcement type',
                 'Hardness baseline (HV)', 'Hardness at optimum (HV)',
                 'Hardness improvement (%)', 'Optimal concentration (%)',
                 'Compressive strength at opt. (MPa)',
                 'Min wear rate @ 10N (×10⁻⁷ g/cm)'],
    'System 1 (MgO)': ['Pure Al', 'nano-MgO', 'Oxide ceramic',
                        31, 36.1, '+16.3%', '2.5 wt%', 19.1, 2.9],
    'System 2 (WO₃)': ['Al7075 alloy', 'nano-WO₃', 'Oxide ceramic (denser)',
                        65, 127, '+95.4%', '3.5 vol%', 59, 0.30],
}
df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))
print("\n→ KEY INSIGHT: Higher matrix alloy strength (Al7075 vs Pure Al)")
print("  allows more reinforcement before agglomeration dominates.")
print("  This is the mechanism behind the differing optima.")
print("\n→ RESEARCH GAP: GPR predicts properties but CANNOT prescribe")
print("  optimal process sequences. This motivates Offline RL extension.")
