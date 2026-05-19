"""
plots.py — All required diagnostic visualisations for Snake RL
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs("plots", exist_ok=True)

CQL  = "#FF6B6B"   # Q-Learning — coral red
CDQN = "#4ECDC4"   # DQN — teal
CBKG = "#0D1117"   # dark background
CGRID= "#1C2128"
CTXT = "#E6EDF3"
CACC = "#F7DC6F"   # accent yellow

def _style():
    plt.rcParams.update({
        "figure.facecolor": CBKG, "axes.facecolor": CGRID,
        "axes.edgecolor": "#30363D", "axes.labelcolor": CTXT,
        "xtick.color": CTXT, "ytick.color": CTXT,
        "text.color": CTXT, "grid.color": "#30363D",
        "grid.alpha": 0.5, "legend.facecolor": "#161B22",
        "legend.edgecolor": "#30363D", "legend.labelcolor": CTXT,
        "font.family": "monospace",
    })

def _ma(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode="valid")


# ── Fig 0: Game diagram (visual explainer) ──────────────────
def plot_game_diagram(save_as="plots/00_snake_env.png"):
    _style()
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=CBKG)
    ax.set_facecolor(CGRID)
    G = 15
    # grid lines
    for i in range(G+1):
        ax.axhline(i, color="#30363D", lw=0.5)
        ax.axvline(i, color="#30363D", lw=0.5)

    # snake body
    snake = [(7,4),(7,5),(7,6),(7,7),(7,8),(8,8),(9,8),(9,7),(9,6),(10,6)]
    for i,(r,c) in enumerate(snake):
        col = "#4ECDC4" if i > 0 else "#A8FF78"
        rect = plt.Rectangle((c,G-1-r), 1, 1, color=col, zorder=3)
        ax.add_patch(rect)
        if i == 0:
            ax.text(c+0.5, G-1-r+0.5, "H", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#0D1117", zorder=4)

    # food
    food = (4, 11)
    circle = plt.Circle((food[1]+0.5, G-1-food[0]+0.5), 0.4, color="#FF6B6B", zorder=3)
    ax.add_patch(circle)
    ax.text(food[1]+0.5, G-1-food[0]+0.5, "✕", ha="center", va="center",
            fontsize=11, color="white", zorder=4)

    # danger arrows
    ax.annotate("", xy=(7+0.5, G-1-6+0.5), xytext=(7+0.5, G-1-7+0.9),
                arrowprops=dict(arrowstyle="->", color="#FF6B6B", lw=2), zorder=5)
    ax.text(7.5+0.7, G-1-6+0.5, "DANGER", color="#FF6B6B", fontsize=7, va="center", zorder=5)

    ax.set_xlim(0, G); ax.set_ylim(0, G)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Snake Environment — 15×15 Grid", color=CTXT, fontsize=12, fontweight="bold", pad=10)

    patches = [
        mpatches.Patch(color="#A8FF78", label="Head"),
        mpatches.Patch(color="#4ECDC4", label="Body"),
        mpatches.Patch(color="#FF6B6B", label="Food (+10)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_as}")


# ── Fig 1: Learning curves (score per episode) ──────────────
def plot_learning_curves(ql_result, dqn_result, window=80,
                         save_as="plots/01_learning_curves.png"):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=CBKG)
    fig.suptitle("Learning Curves — Score & Reward per Episode",
                 color=CTXT, fontsize=13, fontweight="bold")

    for ax, key, ylabel, title in [
        (axes[0], "episode_scores",  "Score (food eaten)", "Score per Episode"),
        (axes[1], "episode_rewards", "Total Reward",       "Reward per Episode"),
    ]:
        for data, color, name in [
            (ql_result[key],  CQL,  "Q-Learning"),
            (dqn_result[key], CDQN, "DQN"),
        ]:
            ax.plot(data, alpha=0.12, color=color, lw=0.7)
            sm = _ma(data, window)
            ax.plot(np.arange(len(sm)), sm, color=color, lw=2, label=f"{name} ({window}-ep avg)")
        ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
        ax.set_title(title, color=CTXT, fontsize=10)
        ax.legend(fontsize=9); ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_as}")


# ── Fig 2: Q-value / value heatmap  ─────────────────────────
def plot_value_heatmap(ql_agent, dqn_agent, save_as="plots/02_value_heatmap.png"):
    """
    For tabular Q-Learning: show max Q over actions for each known state key.
    We enumerate all 2^11 possible states and show max Q.
    For DQN: show max Q-value output for each state.
    Displayed as bar chart sorted by value — captures relative state valuation.
    """
    import torch
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=CBKG)
    fig.suptitle("State Value Estimation — max Q(s, ·) Distribution",
                 color=CTXT, fontsize=13, fontweight="bold")

    # QL: iterate known states
    if ql_agent.q_table:
        vals = [np.max(v) for v in ql_agent.q_table.values()]
        vals_sorted = sorted(vals)
        axes[0].fill_between(range(len(vals_sorted)), vals_sorted,
                             color=CQL, alpha=0.7)
        axes[0].axhline(0, color="white", lw=0.8, ls="--")
        axes[0].set_title("Q-Learning: Sorted State Values", color=CTXT)
        axes[0].set_xlabel("State index (sorted by value)")
        axes[0].set_ylabel("max Q(s, ·)")
        axes[0].text(0.05, 0.95, f"States visited: {len(vals):,}",
                     transform=axes[0].transAxes, color=CACC, fontsize=9, va="top")
    else:
        axes[0].text(0.5, 0.5, "No Q-table data", ha="center", va="center",
                     transform=axes[0].transAxes, color=CTXT)

    # DQN: sample random valid states
    device = dqn_agent.device
    n_states = 512
    rng = np.random.default_rng(42)
    # generate plausible binary states
    states = rng.integers(0, 2, size=(n_states, 11)).astype(np.float32)
    with torch.no_grad():
        t = torch.FloatTensor(states).to(device)
        q_vals = dqn_agent.policy_net(t).cpu().numpy()
    max_q = q_vals.max(axis=1)
    sorted_q = np.sort(max_q)
    axes[1].fill_between(range(len(sorted_q)), sorted_q, color=CDQN, alpha=0.7)
    axes[1].axhline(0, color="white", lw=0.8, ls="--")
    axes[1].set_title("DQN: Sorted Max Q-values (512 sampled states)", color=CTXT)
    axes[1].set_xlabel("State index (sorted by value)")
    axes[1].set_ylabel("max Q(s, ·)")

    fig.tight_layout()
    fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_as}")


# ── Fig 3: Policy visualisation  ────────────────────────────
def plot_policy_grid(ql_agent, dqn_agent, save_as="plots/03_policy_grid.png"):
    """
    Show the greedy action for a representative slice of states.
    We vary danger flags and food direction, fix direction=RIGHT.
    Shows as a colour-coded grid: colour = action chosen.
    """
    import torch
    from environment import N_ACTIONS
    _style()

    ACTION_LABELS = {0:"UP", 1:"RIGHT", 2:"DOWN", 3:"LEFT"}
    ACTION_COLORS = {0:"#A8FF78", 1:"#4ECDC4", 2:"#FF6B6B", 3:"#F7DC6F"}

    # Build a representative set:
    # Rows: food direction (4 combos of food_left/right × food_up/down)
    # Cols: danger combo (8 combos of danger_straight/right/left)
    food_configs = [
        (1,0,1,0,"food ↖"), (1,0,0,1,"food ↙"),
        (0,1,1,0,"food ↗"), (0,1,0,1,"food ↘"),
    ]
    danger_configs = [
        (0,0,0,"no danger"), (1,0,0,"S-danger"), (0,1,0,"R-danger"),
        (0,0,1,"L-danger"),  (1,1,0,"SR-danger"),(1,0,1,"SL-danger"),
        (0,1,1,"RL-danger"), (1,1,1,"all-danger"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), facecolor=CBKG)
    fig.suptitle("Policy Visualisation — Greedy Action for Representative States",
                 color=CTXT, fontsize=13, fontweight="bold")

    for ax, agent, title in [(axes[0], ql_agent, "Q-Learning"),
                              (axes[1], dqn_agent, "DQN")]:
        ax.set_title(title, color=CTXT, fontsize=10)
        grid = np.zeros((len(food_configs), len(danger_configs)))
        for i, (fl,fr,fu,fd,_) in enumerate(food_configs):
            for j, (ds,dr,dl,_) in enumerate(danger_configs):
                # dir=RIGHT (0,1,0,0)
                state = np.array([ds,dr,dl, 0,1,0,0, fl,fr,fu,fd], dtype=np.float32)
                if hasattr(agent, 'q_table'):
                    key = tuple(int(x) for x in state)
                    if key in agent.q_table:
                        a = int(np.argmax(agent.q_table[key]))
                    else:
                        a = 1  # default forward
                else:
                    import torch
                    with torch.no_grad():
                        t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                        a = int(agent.policy_net(t).argmax(dim=1).item())
                grid[i, j] = a

        im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0, vmax=3)
        ax.set_xticks(range(len(danger_configs)))
        ax.set_xticklabels([d[3] for d in danger_configs], rotation=35, fontsize=7)
        ax.set_yticks(range(len(food_configs)))
        ax.set_yticklabels([f[4] for f in food_configs], fontsize=8)

        # annotate with action letter
        for i in range(len(food_configs)):
            for j in range(len(danger_configs)):
                a_int = int(grid[i,j])
                ax.text(j, i, ACTION_LABELS[a_int][0],
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")

    patches = [mpatches.Patch(color=ACTION_COLORS[a], label=ACTION_LABELS[a])
               for a in range(4)]
    axes[1].legend(handles=patches, loc="lower right", fontsize=8,
                   ncol=4, bbox_to_anchor=(1, -0.35))
    fig.tight_layout()
    fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_as}")


# ── Fig 4: Stability plot  ───────────────────────────────────
def plot_stability(ql_scores, dqn_scores, window=80,
                   save_as="plots/04_stability.png"):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=CBKG)
    fig.suptitle("Stability — Score Mean ± Std Across 5 Independent Runs",
                 color=CTXT, fontsize=13, fontweight="bold")

    for ax, all_scores, color, name in [
        (axes[0], ql_scores,  CQL,  "Q-Learning"),
        (axes[1], dqn_scores, CDQN, "DQN"),
    ]:
        smoothed = np.array([_ma(r, window) for r in all_scores])
        mean = smoothed.mean(axis=0)
        std  = smoothed.std(axis=0)
        xs   = np.arange(len(mean))
        ax.plot(xs, mean, color=color, lw=2.5, label=f"{name} mean")
        ax.fill_between(xs, mean-std, mean+std, color=color, alpha=0.25, label="±1 std")
        for run in smoothed:
            ax.plot(xs, run, color=color, alpha=0.2, lw=0.8)
        ax.set_xlabel("Episode"); ax.set_ylabel("Score (food eaten)")
        ax.set_title(f"{name} — {len(all_scores)} runs", color=CTXT)
        ax.legend(fontsize=9); ax.grid(True)
        ax.text(0.05, 0.95, f"Final σ²={np.var(smoothed[:,-1]):.3f}",
                transform=ax.transAxes, color=CACC, fontsize=9, va="top")

    fig.tight_layout()
    fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_as}")


# ── Fig 5: Convergence speed  ────────────────────────────────
def plot_convergence(ql_result, dqn_result, threshold=5, window=80,
                     save_as="plots/05_convergence.png"):
    _style()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CBKG)

    for result, color, name in [
        (ql_result,  CQL,  "Q-Learning"),
        (dqn_result, CDQN, "DQN"),
    ]:
        sc = result["episode_scores"]
        sm = _ma(sc, window)
        xs = np.arange(len(sm))
        ax.plot(xs, sm, color=color, lw=2.5, label=f"{name} ({window}-ep avg)")
        ax.fill_between(xs, 0, sm, color=color, alpha=0.08)
        cross = np.where(sm >= threshold)[0]
        if len(cross):
            ep = cross[0]
            ax.axvline(ep, color=color, lw=1.5, ls=":", alpha=0.9)
            ax.scatter([ep], [sm[ep]], color=color, s=80, zorder=5)
            ax.text(ep+10, sm[ep]+0.1, f"ep {ep}", color=color, fontsize=9)

    ax.axhline(threshold, color=CACC, lw=1.2, ls="--", label=f"Threshold = {threshold}")
    ax.set_xlabel("Episode"); ax.set_ylabel("Score (food eaten, smoothed)")
    ax.set_title("Convergence Speed — Episodes to Reach Score Threshold",
                 color=CTXT, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_as}")


# ── Fig 6: DQN loss curve  ───────────────────────────────────
def plot_loss_curve(dqn_result, window=80, save_as="plots/06_dqn_loss.png"):
    _style()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=CBKG)
    losses = dqn_result["losses"]
    nonzero = losses[losses > 0]
    sm = _ma(nonzero, window)
    ax.plot(nonzero, alpha=0.2, color=CDQN, lw=0.7)
    ax.plot(np.arange(len(sm)), sm, color=CDQN, lw=2, label=f"{window}-ep avg")
    ax.set_xlabel("Episode"); ax.set_ylabel("Huber Loss")
    ax.set_title("DQN Training Loss (Huber / Smooth L1)",
                 color=CTXT, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_as}")


# ── Fig 7: Epsilon decay  ────────────────────────────────────
def plot_epsilon_decay(ql_agent_eps_log, dqn_agent_eps_log,
                       save_as="plots/07_epsilon_decay.png"):
    _style()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=CBKG)
    ax.plot(ql_agent_eps_log, color=CQL, lw=2, label="Q-Learning ε")
    ax.plot(dqn_agent_eps_log, color=CDQN, lw=2, label="DQN ε")
    ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon (ε)")
    ax.set_title("Exploration Decay Schedule",
                 color=CTXT, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_as}")


# ── Fig 8: Final metrics bar chart  ─────────────────────────
def plot_final_metrics(metrics, save_as="plots/08_final_metrics.png"):
    _style()
    names  = list(metrics.keys())
    colors = [CQL, CDQN][:len(names)]

    keys_labels = [
        ("mean_score",  "Mean Score (eval)"),
        ("mean_reward", "Mean Reward (eval)"),
        ("conv_ep",     "Convergence Episode"),
        ("variance",    "Score Variance (σ²)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor=CBKG)
    fig.suptitle("Final Evaluation Metrics",
                 color=CTXT, fontsize=13, fontweight="bold")

    for ax, (k, label) in zip(axes, keys_labels):
        vals = [metrics[n].get(k, 0) for n in names]
        bars = ax.bar(names, vals, color=colors, edgecolor="#30363D", width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02 if bar.get_height() >= 0 else bar.get_height() * 0.95,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=CTXT)
        ax.set_title(label, color=CTXT, fontsize=9)
        ax.set_ylabel(label.split("(")[0].strip())
        ax.tick_params(axis="x", labelrotation=10)
        ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_as}")


# ── Fig 9: Score distribution  ──────────────────────────────
def plot_score_distribution(ql_result, dqn_result,
                            save_as="plots/09_score_distribution.png"):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=CBKG)
    fig.suptitle("Score Distribution — Last 500 Episodes",
                 color=CTXT, fontsize=13, fontweight="bold")

    for ax, result, color, name in [
        (axes[0], ql_result,  CQL,  "Q-Learning"),
        (axes[1], dqn_result, CDQN, "DQN"),
    ]:
        scores = result["episode_scores"][-500:]
        ax.hist(scores, bins=20, color=color, alpha=0.8, edgecolor="#0D1117")
        ax.axvline(scores.mean(), color=CACC, lw=2, ls="--",
                   label=f"Mean = {scores.mean():.1f}")
        ax.set_xlabel("Score (food eaten)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name} — Score Distribution", color=CTXT)
        ax.legend(fontsize=9); ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_as}")
