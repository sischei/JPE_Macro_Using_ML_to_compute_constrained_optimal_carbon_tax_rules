import os
import create_figures_paper as cfp

if __name__ == '__main__':
    cfg = next(c for c in cfp.RUN_CONFIGS if c['fig_prefix'] == 'Figure_6')
    cfp.save_table(
        os.path.join(cfg['run_dir'], 'Table_9_optimal_transfer_parameters.csv'),
        os.path.join(cfg['out_dir'], 'Table_9_optimal_transfer_parameters.csv'),
        3
    )
