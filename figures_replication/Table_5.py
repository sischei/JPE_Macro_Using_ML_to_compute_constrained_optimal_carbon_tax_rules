import os
import create_figures_paper as cfp

if __name__ == '__main__':
    cfg = next(c for c in cfp.RUN_CONFIGS if c['fig_prefix'] == 'Figure_5')
    cfp.save_table(
        os.path.join(cfg['run_dir'], 'Table_5_tax.csv'),
        os.path.join(cfg['out_dir'], 'Table_5_tax.csv'),
        3
    )
