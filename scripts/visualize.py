"""Visualize LLM diagnostic results from ``python -m scripts.run_diagnostics``."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"


def plot_exp_a():
    """PR heatmap: repetitive vs varied, layers × positions."""
    data = np.load(f"{RESULTS_DIR}/exp_a.npz")
    rep_pr = data["rep_pr"]
    var_pr = data["var_pr"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Downsample positions for readability
    step = max(1, rep_pr.shape[1] // 40)
    rep_ds = rep_pr[:, ::step]
    var_ds = var_pr[:, ::step]

    for ax, pr, title in [
        (axes[0], rep_ds, "Repetitive Input"),
        (axes[1], var_ds, "Varied Input"),
    ]:
        im = ax.imshow(pr, aspect="auto", cmap="viridis", vmin=0, vmax=60)
        ax.set_xlabel("Position (downsampled)")
        ax.set_ylabel("Layer")
        ax.set_title(f"{title}\nParticipation Ratio")
        plt.colorbar(im, ax=ax, label="PR")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp_a_pr_heatmap.png", dpi=150)
    print(f"Saved {RESULTS_DIR}/exp_a_pr_heatmap.png")
    plt.close()

    # Segment-wise PR comparison at final layer
    rep_seg = data["rep_seg_pr"]
    var_seg = data["var_seg_pr"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rep_seg, label="Repetitive", marker="o", markersize=3)
    ax.plot(var_seg, label="Varied", marker="s", markersize=3)
    ax.set_xlabel("Segment (sentence)")
    ax.set_ylabel("Mean PR (final layer)")
    ax.set_title("Experiment A: PR at Final Layer Across Segments")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp_a_segment_pr.png", dpi=150)
    print(f"Saved {RESULTS_DIR}/exp_a_segment_pr.png")
    plt.close()


def plot_exp_b():
    """Self-torque: angular displacement vs delay at different layers."""
    data = np.load(f"{RESULTS_DIR}/exp_b.npz")
    delays = data["delays"]

    # Recompute per-layer self-torque
    hs = data["hidden_states_last_layer"]  # only saved last layer
    seq_len = hs.shape[0]

    fig, ax = plt.subplots(figsize=(8, 5))

    mean_disps = []
    for j, d in enumerate(delays):
        disps = []
        for pos in range(d, seq_len):
            h1 = hs[pos]
            h2 = hs[pos - d]
            dot = np.dot(h1, h2)
            n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_sim = np.clip(dot / (n1 * n2), -1.0, 1.0)
                disps.append(np.arccos(cos_sim))
        mean_disps.append(np.mean(disps) if disps else 0)

    ax.plot(delays, mean_disps, "o-", label="Layer 12")
    ax.set_xlabel("Position Delay")
    ax.set_ylabel("Mean Angular Displacement (rad)")
    ax.set_title("Experiment B: Self-Torque at Layer 12")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp_b_self_torque.png", dpi=150)
    print(f"Saved {RESULTS_DIR}/exp_b_self_torque.png")
    plt.close()


def plot_exp_c():
    """PR and entropy time series during generation."""
    data = np.load(f"{RESULTS_DIR}/exp_c.npz")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for label, color in [("bland", "tab:blue"), ("surprising", "tab:orange")]:
        pr = data[f"{label}_pr"]
        entropy = data[f"{label}_entropy"]
        axes[0].plot(pr, label=label, alpha=0.7, color=color)
        axes[1].plot(entropy, label=label, alpha=0.7, color=color)

    axes[0].set_ylabel("Participation Ratio")
    axes[0].set_title("Experiment C: PR During Generation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Next-Token Entropy")
    axes[1].set_xlabel("Generation Step")
    axes[1].set_title("Experiment C: Entropy During Generation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp_c_generation.png", dpi=150)
    print(f"Saved {RESULTS_DIR}/exp_c_generation.png")
    plt.close()


def plot_exp_d():
    """Layer regime profile: resonance/torque/orthogonality by layer."""
    data = np.load(f"{RESULTS_DIR}/exp_d.npz")
    regimes = data["layer_regimes"]  # (n_layers, 5): res, tor, orth, pr, gini

    layers = np.arange(1, len(regimes) + 1)
    res = regimes[:, 0] * 100
    tor = regimes[:, 1] * 100
    orth = regimes[:, 2] * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Stacked bar
    ax1.bar(layers, res, label="Resonance", color="tab:blue", alpha=0.8)
    ax1.bar(layers, tor, bottom=res, label="Torque", color="tab:red", alpha=0.8)
    ax1.bar(layers, orth, bottom=res + tor, label="Orthogonality", color="tab:gray", alpha=0.8)
    ax1.set_ylabel("Fraction (%)")
    ax1.set_title("Experiment D: Regime Profile by Layer")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # PR and Gini
    ax2_twin = ax2.twinx()
    ax2.plot(layers, regimes[:, 3], "o-", label="PR", color="tab:green")
    ax2_twin.plot(layers, regimes[:, 4], "s-", label="Gini", color="tab:purple")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Participation Ratio", color="tab:green")
    ax2_twin.set_ylabel("Gini Coefficient", color="tab:purple")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp_d_layer_profile.png", dpi=150)
    print(f"Saved {RESULTS_DIR}/exp_d_layer_profile.png")
    plt.close()


if __name__ == "__main__":
    plot_exp_a()
    plot_exp_b()
    plot_exp_c()
    plot_exp_d()
    print("\nAll plots generated.")
