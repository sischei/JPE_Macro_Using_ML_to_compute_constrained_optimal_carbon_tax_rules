## What is in the folder
This folder contains the model files to solve the pareto improving linear taxes on cumulative emissions model using DEQN. The pretrained models are saved in the folder `runs/jpe_pseudostate_S_transfers_risk_test/main_result_final`.

## How to solve the model from scratch

To solve the models from scratch proceed as follows:

1. Make sure you have a solution to the business-as-usual (jpe_bau_final) model. As the welfares of the BAU model are the reference point for the pareto improving policies.

2. Then, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```


3. Then run the following command to solve the DEQN model and let it run for about 19322 episodes (note that multiple restarts may be needed to achieve satisfactory convergence):

```
$ python run_deepnet.py MODEL_NAME=jpe_pseudostate_S_transfers_risk_test net=OLG_large_gelu optimizer=OLG_BAU run=OLG_BAU_baseline constants=OLG_BAU variables=OLG_BAU
```
Note, in order for the model to converge well, it might be necessary to initially comment out the loss terms in the file [jpe_pseudostate_S_transfers_risk_test/Equations.py](Equations.py) lines 28-33. After 10-20 episodes interrupt training, uncomment the loss terms of the value functions and continue training using 
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/<RUN_DIR> && python run_deepnet.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```
4. Next, run the following command to create the initial sample used to fit the 40 GP models pointing to the run directory of the DEQN model, solved in step 3:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/<RUN_DIR> && python post_process_GP_init_sampling.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

5. Next run the following code to fit the 40 welfare surrogate models:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/<RUN_DIR> && python post_process_GP_init_fit_parallel.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

6. Then point to the BAU reference welfares and set the settings by changing the user settings in the file in ``DEQN_for_IAM/max_pareto_single_GP_optimized_main.py`` lines 89-96 as follows:

```python

# path BAU welfares
path_welfare_BAU = 'runs/jpe_bau_final/<RUN_DIR>/sim_mean_welfare.csv'

START_BAL = True
FINAL_OPTIMIZATION = True
```

7. run the following command to maximize the planner problem using the 40 welfare surrogate models:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/<RUN_DIR> && python max_pareto_single_GP_optimized_main.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

8. Next, to create results as in Figure 5 and Tables 4,5 and 6, run the following command:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/<RUN_DIR> && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

## How to replicate results in the paper
First note that the pretrained DEQN and GP models are saved in the respective run folder `<PATH_TO_THE_FOLDER>/DEQN_for_IAM/runs/jpe_pseudostate_S_transfers_risk_test/main_result_final`.

First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```

Then change the user settings in ``DEQN_for_IAM/max_pareto_single_GP_optimized_main.py`` lines 94-95 as follows:

```python
# path BAU welfares
path_welfare_BAU = 'runs/jpe_bau_final/final/sim_mean_welfare.csv'

START_BAL = False
FINAL_OPTIMIZATION = True
```

Next, to compute the optimal planner policy, run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/main_result_final && python max_pareto_single_GP_optimized_main.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

Finally, to replicate results of Figure 5, Tables 4,5 and 6 and the first entry of Table 10 run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/main_result_final && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

* The results are saved in the RUN_DIR and in the directory: `DEQN_for_IAM/runs/jpe_pseudostate_S_transfers_risk_test/main_result_final`

## How to replicate the results in the online appendix

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

Next, run the following command to replicate the planner policy:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/appendix_theta60_final && python max_pareto_single_GP_optimized_2taxes_appendix.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

Next, run the following command to replicate results for Figure 2 and Tables 5, 6, and 7 of the online appendix:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_S_transfers_risk_test/appendix_theta60_final && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

