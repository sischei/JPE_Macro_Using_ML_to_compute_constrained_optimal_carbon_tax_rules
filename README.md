# Using Machine Learning to Compute Constrained Optimal Carbon Tax Rules


## Description of programs and datasets used

### Organization of the repository

This Python-based code repository supplements the work of [Felix Kuebler](https://sites.google.com/site/fkubler/), [Simon Scheidegger](https://sites.google.com/site/simonscheidegger), and [Oliver Surbek](https://osurbek.github.io) titled _[Using Machine Learning to Compute Constrained Optimal Carbon Tax Rules](#citation)_ (Kuebler et al.; 2026).

This repository contains two folders
1. ["DEQN_for_IAM"](DEQN_for_IAM) which contains Replication codes for *Section 5 - Results*, where non-stationary integrated assessment models (IAMs) are solved by adopting ["Deep Equilibrium Nets (DEQN)"](https://onlinelibrary.wiley.com/doi/epdf/10.1111/iere.12575) and Gaussian Process Regression to compute constrained optimal carbon tax rules. Notice that the codes provided here complement the Online Appendix of our article, where the formal underpinnings of the code are outlined.
    - How to compute the main results of *Section 5 - Results* is detailed in various readmes under the following links: [business as usual results (Sec 5.1)](DEQN_for_IAM/jpe_bau_final/README.md), [welfare improving results (Sec 5.2)](DEQN_for_IAM/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/README.md),  [pareto-improving results (Sec 5.3)](DEQN_for_IAM/jpe_pseudostate_S_transfers_risk_test/README.md), and [pareto-improving results (Sec 5.4)](DEQN_for_IAM/jpe_pseudostate_linear_transfers_risk_test_implied).
2. ["figures_replication"](figures_replication): Replication routines for plotting all the figures that are presented in the paper.



The content and usage of the generic Deep Equilibrium Nets framework are outlined in the corresponding [README](DEQN_for_IAM/README.md).
      


  
### Replication of the numerical results

* To replicate the results of the article step-by-step, a detailed set of instructions is provided _[here](#Replication)_.
  

    
## Computational requirements

### Software requirements

* We provide implementations that use python 3.9

* The basic dependencies are [Tensorflow==2.x](https://www.tensorflow.org/), [hydra](https://hydra.cc/), [Tensorboard](https://www.tensorflow.org/tensorboard) (for monitoring) and [GPytorch](https://gpytorch.ai/).

* The file ``requirements.txt`` lists the detailed dependencies. Please run "pip install -r requirements.txt" as the first step. See [here](https://pip.pypa.io/en/stable/user_guide/#ensuring-repeatability) for further instructions on creating and using the ``requirements.txt`` file.





### Controlled randomness

The random seed for our computations in *Section 5 - Results* is set at ``JPE_Macro_Using_ML_to_compute_constrained_optimal_carbon_tax_rules/DEQN_for_IAM/config/config.yaml``, line 10.

Further seeds are set directly inside scripts where Gaussian Processes are fitted.

* ``JPE_Macro_Using_ML_to_compute_constrained_optimal_carbon_tax_rules/DEQN_for_IAM/``

### Memory and runtime requirements

* To solve one IAM Surrogate model as discussed in *Section 5 - Results* until full convergence, it requires between 1 and 12 hours on nuvolos. All those models presented in the paper were solved using our [DEQN library](DEQN_for_IAM), which we ran on an 16-core Intel compute node on [https://nuvolos.cloud](https://nuvolos.cloud) with 64GB of RAM, and 50Gb of fast local storage (SSD).

* To fit Gaussian Process Surrogates as discussed in *Section 5 - Results* until full convergence, it requires between 30 minutes and 1 hour on nuvolos. All those models presented in the paper were solved using our [DEQN library](DEQN_for_IAM), which we ran on an 16-core Intel compute node on [https://nuvolos.cloud](https://nuvolos.cloud) with 64GB of RAM, and 50Gb of fast local storage (SSD).






# Replication

* This section provides instructions on how to replicate the numerical results of the article. Note that in this readme, we only provide the basic steps to obtain the main results of the article. Highly granular instructions on how to compute all the main results of the article are provided in readmes in the respective sub-folders.

We provide pretrained model checkpoints to ensure the reproducibility of our results. But in the granular readmes in the model sub-folders, we also provide detailed explanations on how to run the codes from scratch. Note that due to stability concerns, solving a model from scratch may require multiple manual restarts.

* To replicate the figures of the manuscript using the model solutions follw the instructions provided in the folder ["figures_replication"](figures_replication)

* The optimal order of running the computer code to replicate the results in article are as follows. 

  1. First run the instructions ``1. Replication of Section 5.1: Solving the Business-As-Usual Model``
  2. Then run the instructions ``2. Replication of Section 5.2: Welfare-improving Linear Taxes on Cumulative Emissions``
  3. Then run the instructions ``3. Replication of Section 5.3: Pareto-improving Linear Taxes on Cumulative Emissions and Optimal Transfers``
  4. Then run the instructions ``4. Replication of Section 5.4: Pareto-improving Linear Taxes on Cumulative Emissions, Carbon Intensity, and Climate Tipping, and Optimal Transfers``



### 1. Replication of Section 5.1: Solving the Business-As-Usual Model

In this section, we provide the basic instructions on how to compute the results presented in section 5.1 of the article. 

First, go to the following folder:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```

To replicate Table 2 and Figure 2, run the following command:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_bau_final/final && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR

```

* Table 2 reports euler errors for the BAU model and Figure 2 reports economic outcomes of the BAU calibration

* More details are provided in [here](DEQN_for_IAM/jpe_bau_final/README.md).


### 2. Replication of Section 5.2: Welfare-improving Linear Taxes on Cumulative Emissions

In this section, we provide the basic instructions on how to compute the results presented in section 5.2 of the article.

First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```

If you want to replicate the solution to the planner problem presented in the text (Not necessary to replicate figures and tables), edit the file ``DEQN_for_IAM/post_process_GP_optim_pipeline_64.py`` and set ``Fit_GP = False`` and ``Optimize_Planner = True`` in lines 49-50.


Then run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/final && python post_process_GP_optim_pipeline_64.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

Next, to replicate results of Table 3 and Figure 4, please run the following command:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/final && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

Finally, to replicate results of Figure 3 run:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/final && python post_process_contour_replication.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```


* More details are provided in [this readme](DEQN_for_IAM/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/README.md).


### 3. Replication of Section 5.3: Pareto-improving Linear Taxes on Cumulative Emissions and Optimal Transfers

In this section, we provide the basic instructions on how to compute the results presented in section 5.3 of the article.

First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```

Next, to compute the optimal planner policy, edit the file ``DEQN_for_IAM/max_pareto_single_GP_optimized_main.py`` and set ``START_BAL = False`` and ``FINAL_OPTIMIZATION = True`` in lines 94-95.

Then run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/main_result_final && python max_pareto_single_GP_optimized_main.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

Next, to replicate the results of Figure 5, and Tables 4,5 and 6 and the first entry of Table 10 run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/main_result_final && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

* More details are provided in [this readme](DEQN_for_IAM/jpe_pseudostate_S_transfers_risk_test/README.md).


### 4. Replication of Section 5.4: Pareto-improving Linear Taxes on Cumulative Emissions, Carbon Intensity, and Climate Tipping, and Optimal Transfers

In this section, we provide the basic instructions on how to compute the results presented in section 5.4 of the article.

First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```

Next, to compute the optimal planner policy, edit the file ``DEQN_for_IAM/max_pareto_single_GP_optimized_main.py`` and set ``START_BAL = False`` and ``FINAL_OPTIMIZATION = True`` in lines 94-95. Then run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_linear_transfers_risk_test_implied/final && python max_pareto_single_GP_optimized_main.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

Next, to replicate Figure 6, Tables 7,8 and 9 and the second entry of Table 10 run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_linear_transfers_risk_test_implied/final && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```


* More details are provided in [this readme](DEQN_for_IAM/jpe_pseudostate_linear_transfers_risk_test_implied/README.md).


### Replication of Appendix Section D.1 Sensitivity to the Abatement Cost Parameter θ1
First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```
Next, in the file DEQN_for_IAM/jpe_pseudostate_S_transfers_risk_test/Definitions.py change line 12 from 0.7 to 0.6 as follows:

```python

def theta1(state, policy_state=None):
    """ Cost coefficient of carbon mitigation """
    _t = tcomp2t(state, policy_state)
    _theta1 = 0.6
    # _theta1 = 0.7
    return _theta1

```

Next, to compute the optimal planner policy, edit the file ``DEQN_for_IAM/max_pareto_single_GP_optimized_2taxes_appendix.py`` and set ``START_BAL = False`` and ``FINAL_OPTIMIZATION = True`` in lines 95-96. Then run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/appendix_theta60_final && python max_pareto_single_GP_optimized_2taxes_appendix.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

Next, run the following command to replicate Figure 2 and Tables 5, 6, and 7 of the online appendix:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/appendix_theta60_final && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

* More details are provided in [this readme](DEQN_for_IAM/jpe_pseudostate_S_transfers_risk_test/README.md).

### Replication of Appendix Section D.2.1 Business-as-Usual under Wider Carbon-Intensity Shocks
First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```

Then in the file DEQN_for_IAM/jpe_bau_final/Dynamics.py increase the size of the shocks by changing line 13 as follows:

```python

# Line 14 for baseline Model
# shocks_kappa = [-0.03, 0., 0.03]

# Line 14 for Appendix Model
shocks_kappa = [-0.035, 0., 0.035]

```

Next, to replicate figure 3 in the online appendix, run:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_bau_final/high_kappa && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

* More details are provided in [here](DEQN_for_IAM/jpe_bau_final/README.md).

### Replication of Appendix Section D.2.2 Pareto-Improving Policy under Wider Carbon-Intensity Shocks
First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```

Next, to compute the optimal planner policy, edit the file ``DEQN_for_IAM/max_pareto_single_GP_optimized_2taxes_appendix.py`` and set ``START_BAL = False`` and ``FINAL_OPTIMIZATION = True`` in lines 95-96. Then run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test_kappa/appendix_final && python max_pareto_single_GP_optimized_2taxes_appendix.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```
Next, point to the correct BAU welfares by editing line 24 in the file `DEQN_for_IAM/post_process_prepare_figures_ee_tables.py` as follows:
```python
BAU_MODEL_SOLUTION = 'high_kappa' # change to 'high_kappa' for replication of online appendix D.2 

```

Next, run the following command to replicate Figure 4 and Tables 8, 9, and 10 of the online appendix:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test_kappa/appendix_final && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

* More details are provided in [this readme](DEQN_for_IAM/jpe_pseudostate_S_transfers_risk_test/README.md).



## Datasets

We use data from RCP scenarios to calibrate the emissions in our model. The data can be downloaded from the Potsdam Institute for Climate Impact Reasearch from the following [URL](http://www.pik-potsdam.de/~mmalte/rcps/). Note that RCP3PD at PIK corresponds to RCP26. Likewise RCP6 at PIK corresponds to RCP60.

The data is stored in the folder `DEQN_for_IAM/calibration_data/dice_rcp_data.xlsx`.


## Authors

* [Felix Kuebler](https://sites.google.com/site/fkubler/) (the University of Zuerich, Department for Banking and Finance, and Swiss Finance Institute)
* [Simon Scheidegger](https://sites.google.com/site/simonscheidegger) (the University of Lausanne, Department of Economics)
* [Oliver Surbek](https://osurbek.github.io) (the University of Lausanne, Department of Economics)


## Citation

```bibtex
@article{kubler2025using,
  title={Using machine learning to compute constrained optimal carbon tax rules},
  author={K{\"u}bler, Felix and Scheidegger, Simon and Surbek, Oliver},
  journal={arXiv preprint arXiv:2507.01704},
  year={2025}
}
```

## Support

This research was supported in part by the University of Chicago Griffin Applied Economics Incubator, and the Swiss National Science Foundation (SNF), under project ID "Can Economic Policy Mitigate Climate-Change".
