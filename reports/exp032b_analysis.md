# exp032b Analysis (best_val_sharpe)

Source: `experiments\exp032b_summary.csv`  (n_runs=40)

### Per-variant (best_val_sharpe)

| Variant | n | mean | std | IQM | min | max |
|---|---|---|---|---|---|---|
| sym | 10 | +1.871 | 0.223 | +1.852 | +1.577 | +2.339 |
| asym | 10 | +1.681 | 0.100 | +1.673 | +1.552 | +1.824 |
| dsr | 10 | +1.809 | 0.205 | +1.821 | +1.507 | +2.110 |
| pt | 10 | +1.667 | 0.094 | +1.670 | +1.525 | +1.788 |

### Pairwise (row = A, col = B; A vs B)

**Cohen's d** (effect size, > 0 means A > B):

| | sym | asym | dsr | pt |
|---|---|---|---|---|
| sym | - | +1.10 | +0.29 | +1.19 |
| asym | -1.10 | - | -0.79 | +0.15 |
| dsr | -0.29 | +0.79 | - | +0.89 |
| pt | -1.19 | -0.15 | -0.89 | - |

**P(A > B)** via bootstrap:

| | sym | asym | dsr | pt |
|---|---|---|---|---|
| sym | - | 1.00 | 0.75 | 1.00 |
| asym | 0.00 | - | 0.03 | 0.63 |
| dsr | 0.25 | 0.97 | - | 0.98 |
| pt | 0.00 | 0.36 | 0.02 | - |

### Figures

- ![boxplot](exp032b_figures/boxplot_sharpe.png)
- ![bar mean](exp032b_figures/bar_mean_sharpe.png)
- ![Cohen d](exp032b_figures/heatmap_cohens_d.png)
- ![P(A>B)](exp032b_figures/heatmap_prob_a_gt_b.png)
