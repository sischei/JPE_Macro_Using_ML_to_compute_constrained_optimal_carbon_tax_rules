import create_figures_paper as cfp

if __name__ == '__main__':
    cfg = next(c for c in cfp.RUN_CONFIGS if c['fig_prefix'] == 'Figure_4')
    cfp.process_config(cfg)
