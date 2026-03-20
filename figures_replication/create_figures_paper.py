"""
create_figures_paper.py
=======================
Recreate all Figures_Paper content (figures + table CSVs) from the per-run
LOG_DIR CSVs produced by post_process_figures_ee_tables.py.

Run from the DEQN root directory:
    python create_figures_paper.py

No TensorFlow, Parameters, or model imports required — only standard
scientific Python libraries.
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------
MAIN_DIRECTORY = os.getcwd()
FIG_DIR = os.path.join(MAIN_DIRECTORY, 'figs')
APP_DIR = os.path.join(FIG_DIR, 'Appendix')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(APP_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared plot settings (mirror post_process_figures_ee_tables.py)
# ---------------------------------------------------------------------------
plt.ticklabel_format(useOffset=False)
FSIZE    = (9, 6.75)
TS_BEG   = 2015   # fixed start year used in all CSV filenames
TS_END   = 2155   # 2015 + 5 * (N_simulated_epsiode_length_analysis - 1) = 2015 + 5*28

# Y-axis labels for every variable that appears in distribution figures
STATE_LABELS = {
    "Temp_x":      "Temperature (°C)",
    "S_x":         "Carbon Stock (GtC)",
    "Damages_GDP": "Damages (share of final output)",
    "tau_x":       "tax rate (τ)",
}

# ---------------------------------------------------------------------------
# Helper: extract variable name from distribution CSV filename
#   states_distribution_2015-2155_Temp_x.csv  -> "Temp_x"
#   def_distribution_2015-2155_Damages_GDP.csv -> "Damages_GDP"
# ---------------------------------------------------------------------------

def _var_from_csv(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r'^(?:states|def)_distribution_\d+-\d+_(.+)$', stem)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Figure helper: quantile fan-chart (states and definitions)
# ---------------------------------------------------------------------------

def plot_distribution(csv_path, ylabel, out_path):
    """
    Recreate a state/definition fan-chart figure.

    CSV columns: year, range_1pct, range_99pct, q10, q25, q50, q75, q90
    """
    df = pd.read_csv(csv_path)
    year = df['year']

    # (quantile value, linestyle, color, legend label)
    Q_STYLE = [
        (10,  ':',  'red',   '10% quantile'),
        (25,  '--', 'blue',  '25% quantile'),
        (50,  '-',  'black', '50% quantile'),
        (75,  '--', 'blue',  '75% quantile'),
        (90,  ':',  'red',   '90% quantile'),
    ]
    Q_COLS = ['q10', 'q25', 'q50', 'q75', 'q90']

    fig, ax = plt.subplots(figsize=FSIZE)
    ax.fill_between(year, df['range_1pct'], df['range_99pct'],
                    facecolor='tab:gray', alpha=0.3,
                    label='Range of sample paths (1% to 99%)')
    for (q_val, ls, color, lbl), col in zip(Q_STYLE, Q_COLS):
        ax.plot(year, df[col], label=lbl, linestyle=ls, color=color, linewidth=2)

    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_xlim([TS_BEG, TS_END])
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
    #             ncol=3, fontsize=12, frameon=True, edgecolor='black')
    plt.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved : {os.path.relpath(out_path)}")


# ---------------------------------------------------------------------------
# Figure helper: emissions calibration/uncertainty chart
# ---------------------------------------------------------------------------

def plot_emissions(csv_path, out_path, is_bau):
    """
    Recreate the emissions uncertainty figure.

    CSV columns (fixed set):
        year, Emissions, Emissions_1, Emissions_99,
        Emissions_10, Emissions_25, Emissions_50, Emissions_75, Emissions_90
    Followed by:
        BAU run   : 4 RCP scenario columns
        Policy run: emissions_BAU column + 4 RCP columns
                    (emissions_BAU may appear twice due to a merge artefact;
                     duplicates are dropped automatically)

    emissions_BAU is always plotted as a black dashed line with no marker.
    RCP scenarios are plotted with distinct colours and markers by position.
    """
    df = pd.read_csv(csv_path)

    # Drop duplicate columns that arise from the merge in the original script
    # (e.g. emissions_BAU appearing twice → pandas names the second one
    #  emissions_BAU.1).  Keep the first occurrence of every base name.
    seen = set()
    keep = []
    for col in df.columns:
        base = col.split('.')[0]   # strips pandas duplicate suffix (.1, .2 …)
        if base not in seen:
            seen.add(base)
            keep.append(col)
    df = df[keep].rename(columns=lambda c: c.split('.')[0])

    year = df['year']

    # Fixed columns — everything else is a calibration/scenario column
    fixed = {
        'year', 'Emissions',
        'Emissions_1', 'Emissions_99',
        'Emissions_10', 'Emissions_25', 'Emissions_50',
        'Emissions_75', 'Emissions_90',
        'emissions_BAU',          # handled separately below
    }
    rcp_cols = [c for c in df.columns if c not in fixed]

    # Colours / markers / styles for the RCP scenario lines (up to 4)
    COLOR_SEQ  = ['#d62728', '#2ca02c', 'blue', '#ff7f0e']
    MARKER_SEQ = ['o',       's',       '^',    'D'      ]
    STYLE_SEQ  = ['-',       ':',       '-.',   '--'     ]

    plt.figure(figsize=FSIZE)
    ax = plt.gca()

    fill_1_99 = ax.fill_between(
        year, df['Emissions_1'], df['Emissions_99'],
        color='tab:gray', alpha=0.3, zorder=1)

    # --- RCP scenario lines ---
    rcp_handles, rcp_labels = [], []
    for col, color, marker, style in zip(rcp_cols, COLOR_SEQ, MARKER_SEQ, STYLE_SEQ):
        h, = ax.plot(
            year, df[col],
            color=color, linestyle=style,
            marker=marker, markersize=7,
            markerfacecolor='white', markevery=5,
            linewidth=2, zorder=5,
        )
        rcp_handles.append(h)
        rcp_labels.append(col)

    # --- emissions_BAU line (non-BAU runs only): black dashed, no marker ---
    bau_handles, bau_labels = [], []
    if not is_bau and 'emissions_BAU' in df.columns:
        h_bau, = ax.plot(
            year, df['emissions_BAU'],
            color='black', linestyle='--', linewidth=2, zorder=5,
        )
        bau_handles.append(h_bau)
        bau_labels.append('BAU emissions')

    # --- Mean model emissions line: black solid ---
    lw_mean = 2
    l_mean, = ax.plot(year, df['Emissions'], color='black', linewidth=lw_mean, zorder=6)

    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Emissions (GtCO2)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    legend_handles = rcp_handles + bau_handles + [l_mean,             fill_1_99    ]
    legend_labels  = rcp_labels  + bau_labels  + ['Emissions (mean)', '1–99% Range']
    # ax.legend(handles=legend_handles, labels=legend_labels,
    #           loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #           ncol=4, fontsize=12, frameon=True, edgecolor='black')
    plt.legend(handles=legend_handles, labels=legend_labels,loc='upper left', fontsize=12, frameon=True, edgecolor='black')
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.35)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved : {os.path.relpath(out_path)}")


# ---------------------------------------------------------------------------
# Figure helper: welfare gains bar chart
# ---------------------------------------------------------------------------

def plot_welfare_gains(csv_path, out_path):
    """
    Recreate the consumption-equivalent welfare gains bar chart.

    CSV columns: birth_year, welfare_gains, consumption_equivalent_gains
    """
    df = pd.read_csv(csv_path)

    plt.figure(figsize=FSIZE)
    plt.bar(df['birth_year'], df['consumption_equivalent_gains'],
            color='black', width=1.5)
    plt.xlabel('Year entering work force', fontsize=14, fontweight='bold')
    plt.ylabel('Consumption equiv. gains/losses', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved : {os.path.relpath(out_path)}")


# ---------------------------------------------------------------------------
# Figure helper: welfare contour chart
# ---------------------------------------------------------------------------

def plot_contour(csv_path, out_path):
    """
    Recreate the contour plot of welfare using pre-gridded CSV data.
    """
    df = pd.read_csv(csv_path)
    N = int(np.sqrt(len(df)))
    X = df['X'].values.reshape((N, N))
    Y = df['Y'].values.reshape((N, N))
    Z = df['Z'].values.reshape((N, N))
    
    levels_fine = np.linspace(-2.5, -2.42, 6)  
    levels_coarse = np.linspace(-3, -2.5, 5)  
    custom_levels = np.concatenate((levels_coarse, levels_fine[1:]))  
    
    plt.figure(figsize=(8, 6))
    
    contourf = plt.contourf(X, Y, Z, levels=custom_levels, cmap="viridis")
    contour_lines = plt.contour(X, Y, Z, levels=custom_levels, colors='black', linewidths=0.8)
    
    levels_to_label = [lvl for lvl in custom_levels if abs(lvl - (-2.484)) > 0.001]
    
    plt.clabel(contour_lines, levels=levels_to_label, inline=True, fontsize=12, fmt='%.2f')
    
    cbar = plt.colorbar(contourf)
    cbar.set_label("Welfare", fontweight='bold', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    plt.xlabel(r"$\vartheta_0$", fontsize=14, fontweight='bold')
    plt.ylabel(r"$\vartheta_E$", fontsize=14, fontweight='bold')
    plt.tick_params(axis='both', labelsize=12)
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout() 
    
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved : {os.path.relpath(out_path)}")


# ---------------------------------------------------------------------------
# Table helper: read, round, and write a CSV to Figures_Paper
# ---------------------------------------------------------------------------

def save_table(src_path, dst_path, decimals):
    if not os.path.exists(src_path):
        print(f"  MISS  : {src_path}  (skipped)")
        return
    df = pd.read_csv(src_path)
    # Round only numeric columns; leave integer/string columns untouched
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
    df.to_csv(dst_path, index=False)
    print(f"  Saved : {os.path.relpath(dst_path)}  (rounded to {decimals} dp)")


def merge_welfare_table(path_2i, path_4i, dst_path):
    """
    Combine the two single-entry welfare CSVs into one Table 10 CSV:

        ,              2 instruments, 4 instruments
        Welfare gain,  0.42%,         0.47%

    Each source CSV is assumed to have exactly one numeric welfare-gain value.
    Values already formatted as "X.XX%" are kept as-is; plain floats are
    formatted as percentages (multiplied by 100 and suffixed with %).
    """
    def _extract_value(path):
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        # Find the first numeric cell across the whole frame
        for col in df.select_dtypes(include='number').columns:
            val = df[col].dropna().iloc[0] 
            # Values are stored as fractions (e.g. 0.0042 → 0.42%)
            return f"{val * 100:.2f}%"
        # Fall back: look for a string column that already contains '%'
        for col in df.columns:
            cell = str(df[col].dropna().iloc[0])
            if '%' in cell:
                return cell
        return "N/A"

    val_2i = _extract_value(path_2i)
    val_4i = _extract_value(path_4i)

    if val_2i is None and val_4i is None:
        print(f"  MISS  : both welfare source files missing — Table 10 skipped")
        return

    out_df = pd.DataFrame({
        '': ['Welfare gain'],
        '2 instruments': [val_2i if val_2i is not None else 'N/A'],
        '4 instruments': [val_4i if val_4i is not None else 'N/A'],
    })
    out_df.to_csv(dst_path, index=False)
    print(f"  Saved : {os.path.relpath(dst_path)}  (merged welfare table)")


def save_euler_table(src_path, dst_path, decimals):
    """
    Read simulated_euler_discrepancies_des.csv (a .describe() output),
    extract the 4 summary rows (EE Mean, EE 99.9, VF Mean, VF 99.9)
    across all Gen 1–11 columns, and save as a clean 4-row CSV.
    """
    if not os.path.exists(src_path):
        print(f"  MISS  : {src_path}  (skipped)")
        return

    # The describe CSV has the statistic names in its first column (index).
    df = pd.read_csv(src_path, index_col=0)

    # Identify and sort ee_* / v_* columns numerically
    ee_cols = sorted([c for c in df.columns if c.startswith('ee_')],
                     key=lambda x: int(x.split('_')[1]))
    v_cols  = sorted([c for c in df.columns if c.startswith('v_')],
                     key=lambda x: int(x.split('_')[1]))

    mean_row   = df.loc['mean']
    pct_row    = df.loc['99.9%']

    # Build the 4-row summary
    rows = []
    for label, cols, stat_row in [
        ('Rel EE Mean',              ee_cols, mean_row),
        ('Rel EE 99.9',              ee_cols, pct_row),
        ('Rel Value Function Mean',  v_cols,  mean_row),
        ('Rel Value Function 99.9',  v_cols,  pct_row),
    ]:
        row = {'Statistic': label}
        for col in cols:
            gen_num = col.split('_')[1]
            row[f'Gen {gen_num}'] = round(stat_row[col], decimals)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(dst_path, index=False)
    print(f"  Saved : {os.path.relpath(dst_path)}  (euler summary, rounded to {decimals} dp)")


# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------
#
# run_dir        : path (relative to DEQN) to the LOG_DIR for this model run
# is_bau         : True only for the BAU reference run
# fig_prefix     : filename prefix in Figures_Paper (e.g. "Figure_2")
# emissions_name : suffix for the emissions PDF; BAU uses "BAU_Emissions"
# out_dir        : FIG_DIR or APP_DIR
# has_welfare    : whether a welfare_gains CSV/figure exists for this run
# tables         : {source csv name in run_dir: dest csv name in out_dir}
# euler_table    : destination CSV name in out_dir for the Euler table,
#                  sourced from simulated_euler_discrepancies_des.csv
#
RUN_CONFIGS = [
    # ------------------------------------------------------------------
    # BAU baseline  →  Figure 2  +  Table 2
    # ------------------------------------------------------------------
    dict(
        run_dir        = '../DEQN_for_IAM/runs/jpe_bau_final/final',
        is_bau         = True,
        fig_prefix     = 'Figure_2',
        emissions_name = 'BAU_Emissions',
        out_dir        = FIG_DIR,
        has_welfare    = False,
        tables         = {},
        euler_table    = ('Table_2_BAU_euler_discrepancies.csv', 4),
    ),

    # ------------------------------------------------------------------
    # BAU high kappa  →  Appendix Figure 3
    # ------------------------------------------------------------------
    dict(
        run_dir        = '../DEQN_for_IAM/runs/jpe_bau_final/high_kappa',
        is_bau         = True,
        fig_prefix     = 'Appendix_Figure_3',
        emissions_name = 'BAU_Emissions',
        out_dir        = APP_DIR,
        has_welfare    = False,
        tables         = {},
        euler_table    = {},
        plot_vars      = {'Temp_x'},   # only Emissions + Temperature
    ),

    # ------------------------------------------------------------------
    # Constant-S-transfer pension risk  →  Figure 4  +  Table 3
    # ------------------------------------------------------------------
    dict(
        run_dir        = '../DEQN_for_IAM/runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/final',
        is_bau         = False,
        fig_prefix     = 'Figure_4',
        emissions_name = 'Emissions',
        out_dir        = FIG_DIR,
        has_welfare    = True,
        tables         = {
            'Sec_5_2_optimal_tax.csv': ('Sec_5_2_optimal_tax.csv', 3),
        },
        euler_table    = ('Table_3_euler_discrepancies.csv', 4),
    ),

    # ------------------------------------------------------------------
    # S-transfers risk, θ₁ = 0.7 (main)  →  Figure 5  +  Tables 4,5,6,10
    # ------------------------------------------------------------------
    dict(
        run_dir        = '../DEQN_for_IAM/runs/jpe_pseudostate_S_transfers_risk_test/main_result_final',
        is_bau         = False,
        fig_prefix     = 'Figure_5',
        emissions_name = 'Emissions',
        out_dir        = FIG_DIR,
        has_welfare    = True,
        tables         = {
            'Table_5_tax.csv':       ('Table_5_tax.csv',       3),
            'Table_6_transfers.csv': ('Table_6_transfers.csv', 3),
            'social_welfare.csv':    ('Table_10_social_welfare_2_instruments.csv', 4),
        },
        euler_table    = ('Table_4_euler_discrepancies.csv', 4),
    ),

    # ------------------------------------------------------------------
    # S-transfers risk, θ₁ = 0.6 (appendix)  →  App. Figure 2  +  App. Tables 5,6,7
    # ------------------------------------------------------------------
    dict(
        run_dir        = '../DEQN_for_IAM/runs/jpe_pseudostate_S_transfers_risk_test/appendix_theta60_final',
        is_bau         = False,
        fig_prefix     = 'Appendix_Figure_2',
        emissions_name = 'Emissions',
        out_dir        = APP_DIR,
        has_welfare    = True,
        tables         = {
            'Appendix_Table_6_tax.csv':       ('Appendix_Table_6_tax.csv',       3),
            'Appendix_Table_7_transfers.csv': ('Appendix_Table_7_transfers.csv', 3),
        },
        euler_table    = ('Appendix_Table_5_euler_discrepancies_theta_1_60.csv', 4),
    ),

    # ------------------------------------------------------------------
    # S-transfers risk, high κ (appendix)  →  App. Figure 4  +  App. Tables 8,9,10
    # ------------------------------------------------------------------
    dict(
        run_dir        = '../DEQN_for_IAM/runs/jpe_pseudostate_S_transfers_risk_test_kappa/appendix_final',
        is_bau         = False,
        fig_prefix     = 'Appendix_Figure_4',
        emissions_name = 'Emissions',
        out_dir        = APP_DIR,
        has_welfare    = False,
        tables         = {
            'Appendix_Table_9_tax.csv':        ('Appendix_Table_9_tax.csv',        3),
            'Appendix_Table_10_transfers.csv': ('Appendix_Table_10_transfers.csv', 3),
        },
        euler_table    = ('Appendix_Table_8_euler_discrepancies.csv', 4),
        plot_vars      = {'Temp_x'},   # only Emissions + Temperature
    ),

    # ------------------------------------------------------------------
    # Linear transfers (4 instruments)  →  Figure 6  +  Tables 7, 8, 9, 10
    # ------------------------------------------------------------------
    dict(
        run_dir        = '../DEQN_for_IAM/runs/jpe_pseudostate_linear_transfers_risk_test_implied/final',
        is_bau         = False,
        fig_prefix     = 'Figure_6',
        emissions_name = 'Emissions',
        out_dir        = FIG_DIR,
        has_welfare    = True,
        tables         = {
            'Table_8_optimal_tax_parameters.csv':      ('Table_8_optimal_tax_parameters.csv',      3),
            'Table_9_optimal_transfer_parameters.csv': ('Table_9_optimal_transfer_parameters.csv', 3),
            'social_welfare.csv':                      ('Table_10_social_welfare_4_instruments.csv', 4),
        },
        euler_table    = ('Table_7_euler_discrepancies.csv', 4),
    ),
]


# ---------------------------------------------------------------------------
# Main execution logic
# ---------------------------------------------------------------------------

def process_config(cfg):
    run_dir = cfg['run_dir']
    prefix  = cfg['fig_prefix']
    out_dir = cfg['out_dir']

    sep = '=' * 70
    print(f"\n{sep}")
    print(f"  {run_dir}")
    print(sep)

    if not os.path.isdir(run_dir):
        print(f"  WARNING: directory not found — skipping entire config.")
        return

    # Optional allowlist of state/def variables to plot (None = plot all)
    plot_vars = cfg.get('plot_vars', None)

    # ------------------------------------------------------------------ #
    # 1. State distribution figures
    # ------------------------------------------------------------------ #
    for csv_path in sorted(glob.glob(os.path.join(run_dir, 'states_distribution_*.csv'))):
        var = _var_from_csv(csv_path)
        if var and var in STATE_LABELS and (plot_vars is None or var in plot_vars):
            plot_distribution(
                csv_path,
                STATE_LABELS[var],
                os.path.join(out_dir, f'{prefix}_{var}.pdf'),
            )

    # ------------------------------------------------------------------ #
    # 2. Definition distribution figures
    # ------------------------------------------------------------------ #
    for csv_path in sorted(glob.glob(os.path.join(run_dir, 'def_distribution_*.csv'))):
        var = _var_from_csv(csv_path)
        if var and var in STATE_LABELS and (plot_vars is None or var in plot_vars):
            plot_distribution(
                csv_path,
                STATE_LABELS[var],
                os.path.join(out_dir, f'{prefix}_{var}.pdf'),
            )

    # ------------------------------------------------------------------ #
    # 3. Emissions figure
    # ------------------------------------------------------------------ #
    emissions_csv = os.path.join(run_dir, 'climate_mean_simulation_calibration_uncertainty.csv')
    if os.path.exists(emissions_csv):
        plot_emissions(
            emissions_csv,
            os.path.join(out_dir, f'{prefix}_{cfg["emissions_name"]}.pdf'),
            cfg['is_bau'],
        )
    else:
        print(f"  MISS  : {emissions_csv}  (skipped)")

    # ------------------------------------------------------------------ #
    # 4. Welfare gains figure  (non-BAU runs only)
    # ------------------------------------------------------------------ #
    if cfg['has_welfare']:
        welfare_csv = os.path.join(run_dir, 'welfare_gains.csv')
        if os.path.exists(welfare_csv):
            plot_welfare_gains(
                welfare_csv,
                os.path.join(out_dir, f'{prefix}_welfare_gains.pdf'),
            )
        else:
            print(f"  MISS  : {welfare_csv}  (skipped)")

    # ------------------------------------------------------------------ #
    # 5. Welfare contour figure
    # ------------------------------------------------------------------ #
    contour_csv = os.path.join(run_dir, 'Figure_3_contour_welfare_data.csv')
    if os.path.exists(contour_csv):
        plot_contour(
            contour_csv,
            os.path.join(out_dir, 'Figure_3_contour_welfare.pdf')
        )

    # ------------------------------------------------------------------ #
    # 6. Table CSVs
    # ------------------------------------------------------------------ #
    for src_name, (dst_name, decimals) in cfg['tables'].items():
        save_table(
            os.path.join(run_dir, src_name),
            os.path.join(out_dir, dst_name),
            decimals,
        )

    # ------------------------------------------------------------------ #
    # 7. Euler discrepancy table
    # ------------------------------------------------------------------ #
    if cfg.get('euler_table'):
        dst_name, decimals = cfg['euler_table']
        save_euler_table(
            os.path.join(run_dir, 'simulated_euler_discrepancies_des.csv'),
            os.path.join(out_dir, dst_name),
            decimals,
        )

def main():
    for cfg in RUN_CONFIGS:
        process_config(cfg)

    print(f"\n{'='*70}")
    print(f"Done.  All outputs written under {os.path.relpath(FIG_DIR)}/")
    print(f"{'='*70}\n")

    # ---------------------------------------------------------------------------
    # Post-loop: merge the two welfare CSVs into a single Table 10
    # ---------------------------------------------------------------------------
    merge_welfare_table(
        os.path.join(FIG_DIR, 'Table_10_social_welfare_2_instruments.csv'),
        os.path.join(FIG_DIR, 'Table_10_social_welfare_4_instruments.csv'),
        os.path.join(FIG_DIR, 'Table_10_welfare.csv'),
    )

if __name__ == '__main__':
    main()
