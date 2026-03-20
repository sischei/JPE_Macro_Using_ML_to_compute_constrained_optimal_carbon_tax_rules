import os
import create_figures_paper as cfp

if __name__ == '__main__':
    cfg = next(c for c in cfp.RUN_CONFIGS if c['fig_prefix'] == 'Appendix_Figure_2')
    cfp.save_table(
        os.path.join(cfg['run_dir'], 'Appendix_Table_6_tax.csv'),
        os.path.join(cfg['out_dir'], 'Appendix_Table_6_tax.csv'),
        3
    )
