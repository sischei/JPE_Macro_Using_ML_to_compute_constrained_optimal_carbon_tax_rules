## What is in the folder
This folder contains the model files to solve the BAU model using DEQN. The pretrained models are saved in the folder `runs/jpe_bau_final/<RUN_DIR>`.

## How to solve the model from scratch

1. To solve the model from scratch, run the following command to solve the DEQN model and let it run for about 3100 episodes (for example by setting `N_episodes` to 3100 in the file [here](../config/run/OLG_BAU_new.yaml)):

```
$ python run_deepnet.py MODEL_NAME=jpe_bau_final net=OLG_small_gelu optimizer=baseline run=OLG_BAU_baseline constants=OLG_BAU variables=OLG_BAU
```
Note, in order for the model to converge well, it might be necessary to initially comment out the loss terms in the file [jpe_bau_final/Equations.py](Equations.py) lines 27-32. After 10-20 episodes interrupt training, uncomment the loss terms of the value functions and continue training using 
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_bau_final/<RUN_DIR> && python run_deepnet.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

2. Next in order to generate results of the model solution, run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_bau_final/<RUN_DIR> && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```
The results will be saved in the run directory `runs/jpe_bau_final/<RUN_DIR>`. 

## How to replicate results in the paper
First note that the pretrained DEQN and GP models are saved in the respective run folders `<PATH_TO_THE_FOLDER>/runs/jpe_bau_final/final`.

First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```

### Replication of Section 5.1

Next in order to replicate results from Figure 2 and table 2, run:
```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_bau_final/final && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

* The results are saved in the run directory:  `<PATH_TO_THE_FOLDER>/runs/jpe_bau_final/final`



### Replication of Appendix Section D.2.2 Pareto-Improving Policy under Wider Carbon-Intensity Shocks Figure 3
First, make sure you are at the root directory of DEQN by changing path to the following sub-directory:

```
$ cd <PATH to the repository>/DEQN_for_IAM
```

Then in the file DEQN_for_IAM/jpe_bau_final/Dynamics.py change line 13 as follows:

```python

# Line 14 for baseline Model
shocks_kappa = [-0.03, 0., 0.03]

# Line 14 for Appendix Model
shocks_kappa = [-0.035, 0., 0.035]

```

Next, to replicate figure 3 in the appendix, run:

```
$ export USE_CONFIG_FROM_RUN_DIR=runs/jpe_bau_final/high_kappa && python post_process_prepare_figures_ee_tables.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```
* The results are saved in the run directory:  `<PATH_TO_THE_FOLDER>/runs/jpe_bau_final/high_kappa`

