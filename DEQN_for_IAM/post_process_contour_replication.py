import importlib
import pandas as pd
import numpy as np
import tensorflow as tf
import Parameters
import matplotlib.pyplot as plt
import State
import PolicyState
import Definitions
from Graphs import run_episode, do_random_step
import sys, shutil, os
import time
import sampling_module as sm
import post_process_GP_module_64 as GPmodule
import pickle

import torch
import gpytorch


from scipy.interpolate import griddata

tf.get_logger().setLevel('CRITICAL')

MAIN_DIRECTORY = os.getcwd()

output_file = Parameters.LOG_DIR + "/welfare_surrogate_SE.pcl"

print(output_file )
with open(output_file, 'rb') as fd:
    model = pickle.load(fd)
    print("GP model loaded from disk")
    print("-------------------------------------------")
fd.close()

# -------------
# take normalized
# Extract train inputs and targets
train_inputs = model.train_inputs[0]
train_targets = model.train_targets



# Convert tensors to NumPy arrays
inputs_numpy = train_inputs.numpy()   
targets_numpy = train_targets.numpy() 

# --------------------
# unnormalized
# --------------------

# load training data
training_set = pd.read_csv(Parameters.LOG_DIR + "/GP_train_data_raw.csv").values




# define training data for the GP
inputs_numpy = training_set[:,:-40]
targets_numpy = training_set[:,-40:] @ np.repeat(0.025,40)

data = pd.DataFrame(data={
    "theta1": inputs_numpy[:, 0],
    "theta2": inputs_numpy[:, 1],
    "welfare": targets_numpy
})  

# Extract x, y, z values
x = data["theta1"].values  # θ1
y = data["theta2"].values  # θ2
z = data["welfare"].values  # Welfare

# Create a grid
xi = np.linspace(x.min(), x.max(), 100)  # θ1 grid
yi = np.linspace(y.min(), y.max(), 100)  # θ2 grid
X, Y = np.meshgrid(xi, yi)

# Interpolate Z values (welfare)
Z = griddata((x, y), z, (X, Y), method='linear')


# Define custom levels: 
levels_fine = np.linspace(-2.5, -2.42, 6)  # Finer in upper range
levels_coarse = np.linspace(-3, -2.5, 5)  # Coarser in lower range
custom_levels = np.concatenate((levels_coarse, levels_fine[1:]))  # Merge levels


# Define custom levels
levels_fine = np.linspace(-2.5, -2.42, 6)  
levels_coarse = np.linspace(-3, -2.5, 5)  
custom_levels = np.concatenate((levels_coarse, levels_fine[1:]))  

plt.figure(figsize=(8, 6))

# 1. Draw ALL the color fills and ALL the black lines
contourf = plt.contourf(X, Y, Z, levels=custom_levels, cmap="viridis")# viridis
contour_lines = plt.contour(X, Y, Z, levels=custom_levels, colors='black', linewidths=0.8)


# Adjust to avoid overlapping labels
levels_to_label = [lvl for lvl in custom_levels if abs(lvl - (-2.484)) > 0.001]

# Pass the filtered list into the 'levels' argument
plt.clabel(contour_lines, levels=levels_to_label, inline=True, fontsize=12, fmt='%.2f')


cbar = plt.colorbar(contourf)
cbar.set_label("Welfare", fontweight='bold', fontsize=14)
cbar.ax.tick_params(labelsize=12)

plt.xlabel(r"$\vartheta_0$", fontsize=14, fontweight='bold')
plt.ylabel(r"$\vartheta_E$", fontsize=14, fontweight='bold')
plt.tick_params(axis='both', labelsize=12)

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout() 

# Save the gridded data to CSV so it can be replicated later without scipy
df_plot = pd.DataFrame({'X': X.flatten(), 'Y': Y.flatten(), 'Z': Z.flatten()})
df_plot.to_csv(Parameters.LOG_DIR + "/Figure_3_contour_welfare_data.csv", index=False)

plt.savefig(Parameters.LOG_DIR +"/Figure_3_contour_welfare.pdf")

