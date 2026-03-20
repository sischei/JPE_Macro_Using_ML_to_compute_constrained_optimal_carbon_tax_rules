"""
Gaussian Process Active Learning Illustration
==============================================
Creates a 3-panel figure showing how GP posterior improves
as active learning sequentially selects evaluation points.

For Appendix C of the carbon tax paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Plotting style — publication quality, LaTeX-compatible
# ---------------------------------------------------------------------------
rcParams.update({
    "text.usetex": False,          # set True if LaTeX is available
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# ---------------------------------------------------------------------------
# GP kernel and posterior (squared-exponential, noise-free)
# ---------------------------------------------------------------------------
def se_kernel(x1, x2, length_scale=0.6, signal_var=1.0):
    """Squared-exponential (RBF) covariance matrix."""
    sq_dist = (x1[:, None] - x2[None, :]) ** 2
    return signal_var * np.exp(-0.5 * sq_dist / length_scale**2)


def gp_posterior(X_train, y_train, X_test, length_scale=0.6, signal_var=1.0, noise=1e-8):
    """Compute GP posterior mean and variance (noise-free observations)."""
    K = se_kernel(X_train, X_train, length_scale, signal_var) + noise * np.eye(len(X_train))
    K_s = se_kernel(X_train, X_test, length_scale, signal_var)
    K_ss = se_kernel(X_test, X_test, length_scale, signal_var)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    v = np.linalg.solve(L, K_s)

    mu = K_s.T @ alpha
    cov = K_ss - v.T @ v
    var = np.diag(cov)
    return mu, var


# ---------------------------------------------------------------------------
# True function (concave "objective" — thematically matches welfare optimization)
# ---------------------------------------------------------------------------
def true_function(x):
    return -0.08 * (x - 1.8)**2 + 0.25 * np.sin(1.3 * x) + 0.6


# ---------------------------------------------------------------------------
# Active learning: select next point at maximum posterior variance
# ---------------------------------------------------------------------------
def select_next_point(X_train, y_train, X_grid, length_scale=0.6, signal_var=1.0):
    """Variance-based acquisition: pick x with highest posterior variance."""
    _, var = gp_posterior(X_train, y_train, X_grid, length_scale, signal_var)
    idx = np.argmax(var)
    return X_grid[idx]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
np.random.seed(42)

# Domain
x_grid = np.linspace(-2.0, 5.0, 500)
y_true = true_function(x_grid)

# GP hyperparameters (fixed for illustration)
ls = 1.3
sv = 0.4

# Initial 3 training points (bracket the domain, gap in middle)
X_init = np.array([-1.5, 0.3, 4.5])
y_init = true_function(X_init)

# ---------------------------------------------------------------------------
# Sequential active learning
# ---------------------------------------------------------------------------
X_train = X_init.copy()
y_train = y_init.copy()

stages = []  # store (X_train, y_train, new_point_or_None) per panel

# Panel (a): initial 3 points
mu, var = gp_posterior(X_train, y_train, x_grid, ls, sv)
stages.append((X_train.copy(), y_train.copy(), mu.copy(), var.copy(), None))

# Select point 4
x_new = select_next_point(X_train, y_train, x_grid, ls, sv)
X_train = np.append(X_train, x_new)
y_train = np.append(y_train, true_function(x_new))
mu, var = gp_posterior(X_train, y_train, x_grid, ls, sv)
stages.append((X_train.copy(), y_train.copy(), mu.copy(), var.copy(), x_new))

# Select point 5
x_new = select_next_point(X_train, y_train, x_grid, ls, sv)
X_train = np.append(X_train, x_new)
y_train = np.append(y_train, true_function(x_new))
mu, var = gp_posterior(X_train, y_train, x_grid, ls, sv)
stages.append((X_train.copy(), y_train.copy(), mu.copy(), var.copy(), x_new))

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)

panel_labels = [
    r"(a)  Initial design: $n = 3$ points",
    r"(b)  After active learning: $n = 4$",
    r"(c)  After active learning: $n = 5$",
]

col_true = "#333333"
col_mean = "#1f77b4"
col_band = "#1f77b4"
col_pts  = "#d62728"
col_new  = "#2ca02c"

for i, ax in enumerate(axes):
    X_tr, y_tr, mu, var, new_pt = stages[i]
    sd = np.sqrt(var)

    # 95% credible band
    ax.fill_between(x_grid, mu - 1.96 * sd, mu + 1.96 * sd,
                    alpha=0.18, color=col_band, linewidth=0)

    # True function
    ax.plot(x_grid, y_true, "--", color=col_true, linewidth=1.0, label="True function $f(x)$")

    # GP posterior mean
    ax.plot(x_grid, mu, "-", color=col_mean, linewidth=1.4, label="GP posterior mean")

    # Training points
    if i == 0:
        ax.plot(X_tr, y_tr, "o", color=col_pts, markersize=6, zorder=5,
                markeredgecolor="white", markeredgewidth=0.6, label="Observations")
    else:
        # Old points
        ax.plot(X_tr[:-1], y_tr[:-1], "o", color=col_pts, markersize=6, zorder=5,
                markeredgecolor="white", markeredgewidth=0.6, label="Observations")
        # Newly added point (highlighted)
        ax.plot(new_pt, true_function(new_pt), "D", color=col_new, markersize=7.5,
                zorder=6, markeredgecolor="white", markeredgewidth=0.6,
                label="New point (max variance)")

    ax.set_title(panel_labels[i], fontsize=10.5)
    ax.set_xlabel("$x$")
    ax.set_xlim(-1.8, 4.8)
    ax.set_ylim(-1.0, 1.5)

    # Light grid
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.set_axisbelow(True)

axes[0].set_ylabel("$f(x)$")

# Single legend below
handles, labels = axes[2].get_legend_handles_labels()
# Reorder: true fn, mean, observations, new point
fig.legend(handles, labels, loc="lower center", ncol=4,
           frameon=True, fancybox=False, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.06, 1, 1])

# Save
fig.savefig("figs/appendix/Appendix_Figure_1.pdf", format="pdf")
#fig.savefig("figs/appendix/Appendix_Figure_1.png", format="png")
print("Saved: Appendix_Figure_1.pdf and Appendix_Figure_1.png")
print(f"\nActive learning selected points at x = {stages[1][4]:.3f} and x = {stages[2][4]:.3f}")
print(f"Initial points: {X_init}")

plt.show()
