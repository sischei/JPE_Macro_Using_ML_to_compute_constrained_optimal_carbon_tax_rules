## What is in the folder
This folder contains the model files to solve the pareto improving linear taxes on cumulative emissions model using DEQN with higher uncertainty in carbon intensity (epsilon_kappa). The pretrained models are saved in the folder `runs/jpe_pseudostate_S_transfers_risk_test_kappa/appendix_final`.

## How to solve the model from scratch

To solve the models from scratch proceed as follows:

1. Make sure you have a solution to the business-as-usual (jpe_bau_final) model with higher uncertainty in carbon intensity (epsilon_kappa). As the welfares of the BAU model are the reference point for the pareto improving policies.

2. Then, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```


3. You can use the main model checkpoints (runs/jpe_pseudostate_S_transfers_risk_test/main_result_final) copy paste them to a new RUN_DIR and edit the config file `runs/jpe_pseudostate_S_transfers_risk_test_kappa/<RUN_DIR>/.hydra/config.yaml` for a warm start and run the pretrained model for about 1000 episodes as follows:
```
learning_rate: 1.0e-07
MODELNAME: jpe_pseudostate_S_transfers_risk_test_kappa
```
Then train using the warmstart:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test_kappa/<RUN_DIR> && python run_deepnet.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

4. Next, run the following command to create the initial sample used to fit the 40 GP models pointing to the run directory of the DEQN model, solved in step 3:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test_kappa/<RUN_DIR> && python post_process_GP_init_sampling.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

5. Next run the following code to fit the 40 welfare surrogate models:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test_kappa/<RUN_DIR> && python post_process_GP_init_fit_parallel.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

6. Then point to the BAU reference welfares and set the settings by changing the user settings in the file in ``DEQN_for_IAM/max_pareto_single_GP_optimized_2taxes_appendix.py`` lines 89-96 as follows:

```python

# path BAU welfares
path_welfare_BAU = 'runs/jpe_bau_final/<RUN_DIR>/sim_mean_welfare.csv'

START_BAL = True
FINAL_OPTIMIZATION = True
```

7. run the following command to maximize the planner problem using the 40 welfare surrogate models:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test_kappa/<RUN_DIR> && python max_pareto_single_GP_optimized_2taxes_appendix.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

8. Next, to create results, run the following command:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test_kappa/<RUN_DIR> && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

## How to replicate results in the online appendix Section D.2.2 Pareto-Improving Policy under Wider Carbon-Intensity Shocks
First note that the pretrained DEQN and GP models are saved in the respective run folder `<PATH_TO_THE_FOLDER>/DEQN_for_IAM/runs/jpe_pseudostate_S_transfers_risk_test_kappa/appendix_final`.

First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```

Then change the user settings in ``DEQN_for_IAM/max_pareto_single_GP_optimized_2taxes_appendix.py`` lines 89-96 as follows:

```python
# path BAU welfares
path_welfare_BAU = 'runs/jpe_bau_final/high_kappa/sim_mean_welfare.csv'

START_BAL = False
FINAL_OPTIMIZATION = True
```

Next, to compute the optimal planner policy, run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test_kappa/appendix_final && python max_pareto_single_GP_optimized_2taxes_appendix.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```
Next, point to the correct BAU welfares by editing line 24 in the file `DEQN_for_IAM/post_process_prepare_figures_ee_tables.py` as follows:
```python
BAU_MODEL_SOLUTION = 'high_kappa' # change to 'high_kappa' for replication of online appendix D.2 

```
Finally, to replicate results of Figure 4 and Tables 8, 9, and 10 of the online appendix run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test_kappa/appendix_final && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

* The results are saved in the RUN_DIR and in the directory: `DEQN_for_IAM/runs/jpe_pseudostate_S_transfers_risk_test_kappa/appendix_final`
