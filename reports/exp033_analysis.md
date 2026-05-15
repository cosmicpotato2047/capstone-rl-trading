# exp033 Slippage Robustness Analysis

## Per-variant exp033 (best Sharpe, 10 seeds)

variant  n    mean     std     min     max
    sym 10 +1.6581 +0.3008 +1.2268 +2.3261
   asym 10 +1.4784 +0.1015 +1.3053 +1.6604
    dsr 10 +1.5505 +0.2611 +1.1857 +2.0272
     pt 10 +1.4593 +0.0961 +1.3013 +1.6376


## exp032b vs exp033 side-by-side

variant  sharpe_exp032b  sharpe_exp033  delta_sharpe  cohens_d_sharpe  mdd_exp032b  mdd_exp033  delta_mdd
    sym          +1.871         +1.658        -0.213           -0.803       +3.269      +3.471     +0.202
   asym          +1.681         +1.478        -0.203           -2.011       +2.276      +2.283     +0.007
    dsr          +1.809         +1.551        -0.259           -1.102       +4.325      +4.877     +0.552
     pt          +1.667         +1.459        -0.207           -2.182       +2.310      +2.343     +0.034


## Slippage resilience (Sharpe retention)

- sym: 88.6%
- asym: 87.9%
- dsr: 85.7%
- pt: 87.6%

## Cluster preservation

- within-cluster |d| mean: 0.288
- across-cluster |d| mean: 0.630
- ratio: 2.19x

## Figures

- ![Menu 1](exp033_figures/menu1_side_by_side.png)
- ![Menu 2](exp033_figures/menu2_pareto_scatter.png)