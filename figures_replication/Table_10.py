import os
import create_figures_paper as cfp

if __name__ == '__main__':
    # 1. Generate 2-instrument table
    cfg_2 = next(c for c in cfp.RUN_CONFIGS if c['fig_prefix'] == 'Figure_5')
    cfp.save_table(
        os.path.join(cfg_2['run_dir'], 'social_welfare.csv'),
        os.path.join(cfg_2['out_dir'], 'Table_10_social_welfare_2_instruments.csv'),
        4
    )

    # 2. Generate 4-instrument table
    cfg_4 = next(c for c in cfp.RUN_CONFIGS if c['fig_prefix'] == 'Figure_6')
    cfp.save_table(
        os.path.join(cfg_4['run_dir'], 'social_welfare.csv'),
        os.path.join(cfg_4['out_dir'], 'Table_10_social_welfare_4_instruments.csv'),
        4
    )

    # 3. Merge
    cfp.merge_welfare_table(
        os.path.join(cfp.FIG_DIR, 'Table_10_social_welfare_2_instruments.csv'),
        os.path.join(cfp.FIG_DIR, 'Table_10_social_welfare_4_instruments.csv'),
        os.path.join(cfp.FIG_DIR, 'Table_10_welfare.csv'),
    )
