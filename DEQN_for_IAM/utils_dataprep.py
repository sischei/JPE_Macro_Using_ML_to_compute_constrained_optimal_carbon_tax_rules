import pandas as pd
import numpy as np

# Assuming euler_discrepancies_des is your DataFrame
# euler_discrepancies_des = euler_discrepancies.abs().describe([.25, .5, .75, .99, 0.999], include='all')

def extract_stats_and_generate_latex(df):
    """
    Extract mean and 99.9% percentile for ee_ and v_ variables and generate LaTeX table
    """
    
    # Filter columns that start with 'ee_' or 'v_'
    ee_cols = [col for col in df.columns if col.startswith('ee_')]
    v_cols = [col for col in df.columns if col.startswith('v_')]
    
    # Sort columns numerically (assuming format like ee_1, ee_2, etc.)
    ee_cols = sorted(ee_cols, key=lambda x: int(x.split('_')[1]))
    v_cols = sorted(v_cols, key=lambda x: int(x.split('_')[1]))
    
    # Extract mean and 99.9% percentile rows
    mean_row = df.loc['mean']
    percentile_999_row = df.loc['99.9%']
    
    # Create the data structure for the table
    generations = range(1, 12)  # Gen 1 to Gen 11
    
    # Extract values
    ee_means = [mean_row[col] for col in ee_cols]
    ee_999 = [percentile_999_row[col] for col in ee_cols]
    v_means = [mean_row[col] for col in v_cols]
    v_999 = [percentile_999_row[col] for col in v_cols]
    
    # Format numbers to 4 decimal places without scientific notation
    def format_sci(val):
        if val == 0:
            return "0"
        else:
            return f"{val:.4f}"
    
    # Generate LaTeX table
    latex_table = r"""
\begin{tabular}{@{}lccccccccccc@{}}
\hline
\hline
& Gen 1 & Gen 2 & Gen 3 & Gen 4 & Gen 5 & Gen 6 & Gen 7 & Gen 8 & Gen 9 & Gen 10 & Gen 11 \\
\hline
"""
    
    # Add Euler Error Mean row
    latex_table += "Rel EE Mean          "
    for val in ee_means:
        latex_table += f"& {format_sci(val)} "
    latex_table += "\\\\\n"
    
    # Add Euler Error 99.9%ile row
    latex_table += "Rel EE 99.9     "
    for val in ee_999:
        latex_table += f"& {format_sci(val)} "
    latex_table += "\\\\\n"
    
    # Add Value Function Mean row
    latex_table += "Rel Value Function Mean       "
    for val in v_means:
        latex_table += f"& {format_sci(val)} "
    latex_table += "\\\\\n"
    
    # Add Value Function 99.9%ile row
    latex_table += "Rel Value Function 99.9  "
    for val in v_999:
        latex_table += f"& {format_sci(val)} "
    latex_table += "\\\\\n"
    
    latex_table += r"""\bottomrule
\end{tabular}"""
# \end{adjustbox}
# \end{table}"""
    
    return latex_table

# Usage example:
# latex_output = extract_stats_and_generate_latex(euler_discrepancies_des)
# print(latex_output)

# Alternative: Create a summary DataFrame first for inspection
def create_summary_dataframe(df):
    """
    Create a summary DataFrame before generating LaTeX
    """
    # Filter columns
    ee_cols = sorted([col for col in df.columns if col.startswith('ee_')], 
                     key=lambda x: int(x.split('_')[1]))
    v_cols = sorted([col for col in df.columns if col.startswith('v_')], 
                    key=lambda x: int(x.split('_')[1]))
    
    # Extract statistics
    mean_row = df.loc['mean']
    percentile_999_row = df.loc['99.9%']
    
    # Create summary DataFrame
    summary_data = {
        'Statistic': ['Rel EE Mean', 'Rel EE 99.9', 
                     'Rel Value Function Mean', 'Rel Value Function 99.9']
    }
    
    # Add data for each generation
    for i in range(1, 12):
        gen_col = f'Gen {i}'
        ee_col = f'ee_{i}'
        v_col = f'v_{i}'
        
        summary_data[gen_col] = [
            mean_row[ee_col] if ee_col in mean_row.index else np.nan,
            percentile_999_row[ee_col] if ee_col in percentile_999_row.index else np.nan,
            mean_row[v_col] if v_col in mean_row.index else np.nan,
            percentile_999_row[v_col] if v_col in percentile_999_row.index else np.nan
        ]
    
    return pd.DataFrame(summary_data)
# Example usage:
# summary_df = create_summary_dataframe(euler_discrepancies_des)
# print(summary_df)
# latex_output = extract_stats_and_generate_latex(euler_discrepancies_des)
# print(latex_output)


# Save to .tex file
def save_latex_table_to_file(df, filename="neural_network_errors_table.tex"):
    """
    Generate LaTeX table and save it to a .tex file
    """
    latex_output = extract_stats_and_generate_latex(df)
    
    with open(filename, 'w') as f:
        f.write(latex_output)
    
    print(f"LaTeX table saved to {filename}")
    return latex_output
# Usage:
# save_latex_table_to_file(euler_discrepancies_des, "my_table.tex")
# 
# Or with default filename:
# save_latex_table_to_file(euler_discrepancies_des)

def write_optimal_tax_latex(intercept, slope, filename):
    """
    Generate LaTeX table for optimal tax parameters and save to file.
    """
    latex_table = r"""\begin{tabular}{lcc}
\hline\hline
Coefficient & Symbol & Value \\
\hline
Intercept & $\vartheta_0$ & %.3f \\
Slope & $\vartheta_E$ & %.3f \\
\hline
\end{tabular}""" % (intercept, slope)
    with open(filename, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {filename}")
    return latex_table

def write_optimal_tax_4_instruments_latex(intercept, slope1, slope2, slope3, filename):
    """
    Generate LaTeX table for optimal tax parameters and save to file.
    """
    latex_table = r"""\begin{tabular}{lcccc}
\hline\hline
Coefficient & Symbol & Value \\
\hline
Intercept & $\vartheta_0$ & %.3f \\
Cumulative emissions & $\vartheta_E$ & %.3f \\
Carbon intensity  & $\vartheta_{\kappa}$ & %.3f \\
Distance to tipping & $\vartheta_{TP}$ & %.3f \\
\hline
\end{tabular}""" % (intercept, slope1, slope2, slope3)
    with open(filename, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {filename}")
    return latex_table

def write_optimal_transfers_latex(transfers, filename):
    """
    Generate LaTeX table for 12 optimal transfer share parameters and save to file.
    """
    assert len(transfers) == 12, "Expected exactly 12 transfer parameters."
    
    latex_table = r"""\begin{tabular}{ccccccccccccc}
\hline
\hline
\textbf{$\vartheta_1$} & \textbf{$\vartheta_2$} & \textbf{$\vartheta_3$} & \textbf{$\vartheta_4$} & \textbf{$\vartheta_5$} & \textbf{$\vartheta_6$} & \textbf{$\vartheta_7$} & \textbf{$\vartheta_8$} & \textbf{$\vartheta_9$} & \textbf{$\vartheta_{10}$} & \textbf{$\vartheta_{11}$} & \textbf{$\vartheta_{12}$} \\
\hline
"""
    # Format each transfer to 3 decimal places
    formatted_transfers = ["%.3f" % float(t) for t in transfers]
    latex_table += " & ".join(formatted_transfers) + r" \\" + "\n"
    latex_table += r"""\hline
\end{tabular}"""

    with open(filename, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {filename}")
    return latex_table
