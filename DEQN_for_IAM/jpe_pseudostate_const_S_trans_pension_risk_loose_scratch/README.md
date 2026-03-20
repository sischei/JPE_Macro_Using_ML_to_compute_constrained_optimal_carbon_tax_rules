## What is in the folder
This folder contains the model files to solve the welfare improving linear taxes on cumulative emissions model using DEQN. The pretrained models are saved in the folder `runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/final`.

## How to solve the model from scratch

To solve the models from scratch proceed as follows:

1. Make sure you have a solution to the business-as-usual (jpe_bau_final) model. As the welfares of the BAU model are the reference point for the welfare improving policies.

2. First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```


3. Then run the following command to solve the DEQN model and let it run for 4311 episodes:


```
$ python run_deepnet.py MODEL_NAME=jpe_pseudostate_const_S_trans_pension_risk_loose_scratch net=OLG_BAU_gelu optimizer=OLG_BAU_step run=OLG_BAU_baseline constants=OLG_BAU variables=OLG_BAU
```

4. Next, run the following command to create the initial sample used to fit the GP model pointing to the run directory of the DEQN model, solved in step 3:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/<RUN_DIR> && python post_process_GP_init_sampling.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

5. Then point to the BAU reference welfares by changing the user settings in the file `DEQN_for_IAM/post_process_GP_optim_pipeline_64.py` lines 45-51 as follows:

```python
# set the path to the BAU welfares
path_welfare_BAU = 'runs/jpe_bau_final/<BAU_RUN_DIR>/sim_mean_welfare.csv'


# Run settings
Fit_GP = True
Optimize_Planner = True
Calculate_LOO = False 

```

6. Then run the following command to fit the GP model and solve the planner problem:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/<RUN_DIR> && python post_process_GP_optim_pipeline_64.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

7. Next, to replicate Table 3 and Figure 4, run the following command:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/<RUN_DIR> && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

## How to replicate results in the paper
First note that the pretrained DEQN and GP models are saved in the respective run folder `<PATH_TO_THE_FOLDER>/DEQN_for_IAM/runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/final`.

First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```


If you want to replicate the solution to the planner problem presented in the text (Not necessary to replicate figures and tables), run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/final && python post_process_GP_optim_pipeline_64.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

Next, to replicate Table 3 and Figure 4, please run the following command:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/final && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

To replicate Figure 3 run:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/final && python post_process_contour_replication.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

* The results are saved in the RUN_DIR: `runs/jpe_pseudostate_const_S_trans_pension_risk_loose_scratch/final`


