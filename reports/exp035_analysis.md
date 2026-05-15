# exp035 Test (out-of-sample) Analysis

Source: `experiments/exp035_summary.csv`


## ATR Baseline Test: Sharpe -0.055, Return -0.98%, MDD 8.04% (Trades 1750, Cycles 824)


## Per source x variant

 source variant  n    mean     std     iqm  cvar_5     min     max  t_stat     p_t
exp032b     sym 10 +0.0902 +0.1114 +0.1003 -0.1268 -0.1268 +0.2342 +2.5598 +0.0153
exp032b    asym 10 +0.1733 +0.1997 +0.1578 -0.0824 -0.0824 +0.4865 +2.7444 +0.0113
exp032b     dsr 10 -0.1222 +0.1936 -0.1260 -0.4676 -0.4676 +0.2001 -1.9951 +0.9614
exp032b      pt 10 +0.3667 +0.2879 +0.3270 +0.0344 +0.0344 +0.8681 +4.0271 +0.0015
 exp034     sym 15 +0.0007 +0.1941 -0.0135 -0.2828 -0.2828 +0.3874 +0.0141 +0.4945
 exp034    asym 15 +0.1749 +0.2509 +0.1273 -0.1435 -0.1435 +0.6933 +2.6994 +0.0086
 exp034     dsr 15 +0.0703 +0.2534 +0.0922 -0.4282 -0.4282 +0.4148 +1.0742 +0.1504
 exp034      pt 15 +0.3389 +0.3084 +0.3085 -0.0465 -0.0465 +1.0382 +4.2559 +0.0004


## Val vs Test

variant  exp032b_val  exp032b_test  exp032b_gap  exp034_val  exp034_test  exp034_gap
    sym      +1.8709       +0.0902      -1.7807     +1.3020      +0.0007     -1.3013
   asym      +1.6813       +0.1733      -1.5079     +1.0433      +0.1749     -0.8684
    dsr      +1.8092       -0.1222      -1.9313     +1.4129      +0.0703     -1.3426
     pt      +1.6665       +0.3667      -1.2999     +1.0925      +0.3389     -0.7536


## Cluster preservation on Test

- exp032b: within 1.062 | across 1.319 | ratio 1.24x
- exp034: within 0.446 | across 0.864 | ratio 1.94x

## Figures

- ![Menu 2](exp035_figures/menu2_val_vs_test.png)
- ![Menu 3](exp035_figures/menu3_boxplot.png)