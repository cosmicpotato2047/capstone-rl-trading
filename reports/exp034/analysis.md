# exp034 CPCV Analysis

Source: `experiments/exp034_summary.csv` (60 runs)


## Per-variant DSR

variant  n_paths  sr_mean  sr_std     iqm  cvar_5  sr_min  sr_max   t_stat     p_t  sr_star        dsr_z   dsr_p
    sym       15  +1.3020 +0.4752 +1.2815 +0.5794 +0.5794 +2.0506 +10.6120 +0.0000  +0.2043     +14.9552 +0.0000
   asym       15  +1.0433 +0.4742 +0.9535 +0.5031 +0.5031 +1.9231  +8.5209 +0.0000  +0.2039 +209448.8775 +0.0000
    dsr       15  +1.4129 +0.3777 +1.4334 +0.8901 +0.8901 +1.8973 +14.4892 +0.0000  +0.1624     +17.3198 +0.0000
     pt       15  +1.0925 +0.5104 +1.0099 +0.5028 +0.5028 +2.0433  +8.2909 +0.0000  +0.2194 +202422.4694 +0.0000


## Cluster preservation

- within-cluster |d| mean: 0.179
- across-cluster |d| mean: 0.636
- ratio: 3.55x  (reference: exp032b 2.22x, exp033 2.19x)

## Figures

- ![heatmap](exp034_figures/menu2_heatmap.png)
- ![boxplot](exp034_figures/menu3_boxplot.png)
- ![DSR table](exp034_figures/menu4_dsr_table.png)