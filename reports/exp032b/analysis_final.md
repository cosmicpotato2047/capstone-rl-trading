# exp032b Analysis (final_val_sharpe)

Source: `experiments\exp032b_summary.csv`  (n_runs=40)

### Per-variant (final_val_sharpe)

| Variant | n | mean | std | IQM | min | max |
|---|---|---|---|---|---|---|
| sym | 10 | +1.015 | 0.433 | +0.987 | +0.462 | +1.753 |
| asym | 10 | +1.101 | 0.265 | +1.126 | +0.567 | +1.469 |
| dsr | 10 | +1.204 | 0.410 | +1.144 | +0.757 | +1.991 |
| pt | 10 | +1.082 | 0.178 | +1.103 | +0.787 | +1.378 |

### Pairwise (row = A, col = B; A vs B)

**Cohen's d** (effect size, > 0 means A > B):

| | sym | asym | dsr | pt |
|---|---|---|---|---|
| sym | - | -0.24 | -0.45 | -0.20 |
| asym | +0.24 | - | -0.30 | +0.09 |
| dsr | +0.45 | +0.30 | - | +0.39 |
| pt | +0.20 | -0.09 | -0.39 | - |

**P(A > B)** via bootstrap:

| | sym | asym | dsr | pt |
|---|---|---|---|---|
| sym | - | 0.28 | 0.14 | 0.31 |
| asym | 0.72 | - | 0.24 | 0.59 |
| dsr | 0.85 | 0.74 | - | 0.81 |
| pt | 0.68 | 0.42 | 0.18 | - |

### Figures

- ![boxplot](exp032b_figures_final/boxplot_sharpe.png)
- ![bar mean](exp032b_figures_final/bar_mean_sharpe.png)
- ![Cohen d](exp032b_figures_final/heatmap_cohens_d.png)
- ![P(A>B)](exp032b_figures_final/heatmap_prob_a_gt_b.png)
