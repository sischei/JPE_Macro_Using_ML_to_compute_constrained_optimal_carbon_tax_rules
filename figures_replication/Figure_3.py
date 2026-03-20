import os
import create_figures_paper as cfp

if __name__ == '__main__':
    # Figure 3 (contour) uses the const_S_trans_pension_risk_loose_scratch config 
    # which is stored under the Figure_4 configuration prefix.
    cfg = next(c for c in cfp.RUN_CONFIGS if c['fig_prefix'] == 'Figure_4')
    contour_csv = os.path.join(cfg['run_dir'], 'Figure_3_contour_welfare_data.csv')
    if os.path.exists(contour_csv):
        cfp.plot_contour(contour_csv, os.path.join(cfg['out_dir'], 'Figure_3_contour_welfare.pdf'))
    else:
        print(f"  MISS  : {contour_csv} (skipped)")
