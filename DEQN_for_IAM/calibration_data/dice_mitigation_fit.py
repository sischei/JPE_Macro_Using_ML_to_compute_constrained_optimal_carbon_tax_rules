# ------------------------------------------------------------------------------
# This script fits a curve to the data points of the dice optimal mitigation for the first 100 years
# where mu is a function of time. The curve is a polynomial of degree 2.
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load data from calibration data
path = 'mitigation_paths.csv'
mu_data = pd.read_csv(path)

# Parameters for the polynomial fit
# I fit the first 110 observations (results are not sensitive to this choice)
timestep = 5 # one period corresponds to 5 years
n_obs_to_fit = 110
n_poly = 2
# set time period
x = np.arange(n_obs_to_fit)
x = x/timestep

# set mu values
y = mu_data['mu_dice2016'].values[:n_obs_to_fit]

polin = np.polyfit(x,y, n_poly)
print("polynomials: ",polin)
poly_model = np.poly1d(polin)

# Evaluate the polynomial
y_pred = poly_model(x)
plt.plot(x,y, label="data")
plt.plot(x,y_pred, label="pred")
plt.legend()
plt.savefig('dice_mitigation_fit.png')
plt.close()


# --------------------------------------------------------------------------- #
# second exercise: try to match RCP 4.5
# --------------------------------------------------------------------------- #

# load data from calibration data
df = pd.read_excel('dice_rcp_data.xlsx', skiprows=1)


timestep = 5 # one period corresponds to 5 years
n_obs_to_fit = 152
n_poly = 4
# set time period
x = np.arange(n_obs_to_fit) 
#x = x/timestep
x = x[::timestep]
y = df['RCP4.5(GtC)'].values[:n_obs_to_fit] 

# get every 5th value of y
y = y[::timestep]
polin = np.polyfit(x,y, n_poly)
print("polynomials: ",polin)
poly_model = np.poly1d(polin)

# Evaluate the polynomial
y_pred = poly_model(x)
plt.plot(x,y, label="data")
plt.plot(x,y_pred, label="pred")
plt.legend()
plt.close()

y_pred_scaled = y_pred/100

y_pred_scaled_growth = y_pred_scaled[1:]/y_pred_scaled[:-1] -1

rho_t = 1 + y_pred_scaled_growth

# append linear decay to rho_t
# last_diff = rho_t[-1] - rho_t[-2]
# for i in range(100):
#     rho_t = np.append(rho_t, rho_t[-1] + last_diff)

rho_t = rho_t[:20]

for i in range(200):
    rho_t = np.append(rho_t, rho_t[-1]*0.97)
# redo with actual y
y_scaled = y/100
y_scaled_growth = y_scaled[1:]/y_scaled[:-1] 
#rho_t = y_scaled_growth

#rho_t = np.append(rho_t, np.repeat(rho_t[-1],100))

shocks = [-0.01,0.00,0.01]

T = int(150/timestep) +30


kappa0 = 0.3502
N = 5000
kappa_df = pd.DataFrame(columns=range(N))

for n in range(N):
    kappa_series = [kappa0]
    for t in range(T):
        if kappa_series[t] > 0.0:
            kappa = kappa_series[t] * rho_t[t] + np.random.choice(shocks)
            if kappa < 0:
                kappa = 0
        else:
            kappa = 0

        kappa_series.append(kappa)

    kappa_df[n] = kappa_series

kappa_df.plot(legend=False)
plt.show()
plt.close()

plt.plot(np.mean(kappa_df.to_numpy(),1))
