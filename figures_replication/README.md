# Replication of the figures and tables of the manuscript "Using Machine Learning to Compute Constrained Optimal Carbon Tax Rules"

- Each script generates a figure or a table of the manuscript. 
- The data used to generate the figures is produced from the solution of the models. Replication code for the solutions can be found in the folder [DEQN_for_IAM](../DEQN_for_IAM).
- The figures and tables are saved in the folder [figs](figs).

To replicate the figures 1-6 and tables 2-10 in the main text, as well as, figures 2-4 and tables 5-10 of the online appendix please make sure you are in the folder /figures_replication and do one of the following:

Either run the script that generates all figures and tables in one shot:

```
$ python create_figures_paper.py
```

Then to replicate Appendix Figure 1 run: 
```
$ python Appendix_figure_1.py
```


Alternatively run the script for a specific figure or table, for example Figure 1 can be replicated by running:
```
$ python Appendix_figure_1.py
```


