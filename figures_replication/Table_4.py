import os
import create_figures_paper as cfp

if __name__ == '__main__':
    cfg = next(c for c in cfp.RUN_CONFIGS if c['fig_prefix'] == 'Figure_5')
    dst, decimals = cfg['euler_table']
    cfp.save_euler_table(
        os.path.join(cfg['run_dir'], 'simulated_euler_discrepancies_des.csv'),
        os.path.join(cfg['out_dir'], dst),
        decimals
    )
