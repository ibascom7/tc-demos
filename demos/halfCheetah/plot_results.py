import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

num_seeds = 3
eval_freq = 5000
env_name = "HalfCheetah-v5"

def load_evals(prefix):
	all_evals = []
	for seed in range(num_seeds):
		path = os.path.join(RESULTS_DIR, f"{prefix}_{env_name}_{seed}.npy")
		evals = np.load(path)
		all_evals.append(evals)
	min_len = min(len(e) for e in all_evals)
	return np.array([e[:min_len] for e in all_evals])

td3_evals = load_evals("TD3")
ddpg_evals = load_evals("DDPG")

# Use the shorter length for shared x-axis
min_len = min(td3_evals.shape[1], ddpg_evals.shape[1])
td3_evals = td3_evals[:, :min_len]
ddpg_evals = ddpg_evals[:, :min_len]
timesteps = np.arange(min_len) * eval_freq

td3_mean = td3_evals.mean(axis=0)
td3_std = td3_evals.std(axis=0)
ddpg_mean = ddpg_evals.mean(axis=0)
ddpg_std = ddpg_evals.std(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: TD3 vs DDPG mean ± std
ax = axes[0]
ax.plot(timesteps, td3_mean, color="tab:blue", linewidth=2, label="TD3")
ax.fill_between(timesteps, td3_mean - td3_std, td3_mean + td3_std, alpha=0.2, color="tab:blue")
ax.plot(timesteps, ddpg_mean, color="tab:orange", linewidth=2, label="DDPG")
ax.fill_between(timesteps, ddpg_mean - ddpg_std, ddpg_mean + ddpg_std, alpha=0.2, color="tab:orange")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Average Eval Reward")
ax.set_title(f"TD3 vs DDPG on {env_name} (mean ± std, {num_seeds} seeds)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Per-seed learning curves
ax = axes[1]
for seed in range(num_seeds):
	ax.plot(timesteps, td3_evals[seed], label=f"TD3 seed {seed}", linewidth=1.5, linestyle="-")
	ax.plot(timesteps, ddpg_evals[seed], label=f"DDPG seed {seed}", linewidth=1.5, linestyle="--")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Average Eval Reward")
ax.set_title(f"TD3 vs DDPG on {env_name} (per seed)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Final performance bar chart
ax = axes[2]
td3_final = td3_evals[:, -1]
ddpg_final = ddpg_evals[:, -1]
x = np.arange(num_seeds)
width = 0.35
ax.bar(x - width/2, td3_final, width, label="TD3", color="tab:blue")
ax.bar(x + width/2, ddpg_final, width, label="DDPG", color="tab:orange")
ax.set_xticks(x)
ax.set_xticklabels([f"Seed {i}" for i in range(num_seeds)])
ax.set_ylabel("Final Eval Reward")
ax.set_title("Final Performance Comparison")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_results.png"), dpi=150)
plt.show()

print(f"\nResults summary:")
print(f"  TD3  — Mean final reward: {td3_final.mean():.1f} ± {td3_final.std():.1f}")
print(f"  DDPG — Mean final reward: {ddpg_final.mean():.1f} ± {ddpg_final.std():.1f}")
print(f"  Plot saved to {os.path.join(RESULTS_DIR, 'training_results.png')}")
