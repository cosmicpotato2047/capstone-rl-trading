# Phase 15 - Significance + Distribution Shift + Three-env figures

## A) Bootstrap P(RL > ATR) per (variant, environment)

variant                  env  n    mean     atr  p_gt_atr  ci95_low  ci95_high                                             atr_ref
    sym   Val (single-split) 10 +1.8709 +1.5050   +1.0000   +1.7461    +2.0093                                           Val 1.505
    sym       Slippage 0.02% 10 +1.6581 +1.5050   +0.9656   +1.4933    +1.8464 Val 1.505 (caveat: ATR not retrained with slippage)
    sym CPCV (15 paths test) 15 +1.3020 +1.5050   +0.0480   +1.0786    +1.5396                Val 1.505 (caveat: ATR not per-path)
    sym  Test (OOS, exp032b) 10 +0.0902 -0.0550   +1.0000   +0.0211    +0.1525                                         Test -0.055
    sym   Test (OOS, exp034) 15 +0.0007 -0.0550   +0.8775   -0.0904    +0.0992                                         Test -0.055
   asym   Val (single-split) 10 +1.6813 +1.5050   +1.0000   +1.6245    +1.7426                                           Val 1.505
   asym       Slippage 0.02% 10 +1.4784 +1.5050   +0.1901   +1.4195    +1.5386 Val 1.505 (caveat: ATR not retrained with slippage)
   asym CPCV (15 paths test) 15 +1.0433 +1.5050   +0.0001   +0.8261    +1.2842                Val 1.505 (caveat: ATR not per-path)
   asym  Test (OOS, exp032b) 10 +0.1733 -0.0550   +1.0000   +0.0610    +0.2957                                         Test -0.055
   asym   Test (OOS, exp034) 15 +0.1749 -0.0550   +1.0000   +0.0604    +0.3007                                         Test -0.055
    dsr   Val (single-split) 10 +1.8092 +1.5050   +1.0000   +1.6876    +1.9272                                           Val 1.505
    dsr       Slippage 0.02% 10 +1.5505 +1.5050   +0.7087   +1.4014    +1.7068 Val 1.505 (caveat: ATR not retrained with slippage)
    dsr CPCV (15 paths test) 15 +1.4129 +1.5050   +0.1680   +1.2253    +1.5907                Val 1.505 (caveat: ATR not per-path)
    dsr  Test (OOS, exp032b) 10 -0.1222 -0.0550   +0.1254   -0.2374    -0.0097                                         Test -0.055
    dsr   Test (OOS, exp034) 15 +0.0703 -0.0550   +0.9734   -0.0565    +0.1910                                         Test -0.055
     pt   Val (single-split) 10 +1.6665 +1.5050   +1.0000   +1.6113    +1.7196                                           Val 1.505
     pt       Slippage 0.02% 10 +1.4593 +1.5050   +0.0598   +1.4041    +1.5170 Val 1.505 (caveat: ATR not retrained with slippage)
     pt CPCV (15 paths test) 15 +1.0925 +1.5050   +0.0010   +0.8576    +1.3492                Val 1.505 (caveat: ATR not per-path)
     pt  Test (OOS, exp032b) 10 +0.3667 -0.0550   +1.0000   +0.2038    +0.5414                                         Test -0.055
     pt   Test (OOS, exp034) 15 +0.3389 -0.0550   +1.0000   +0.1964    +0.4937                                         Test -0.055


## B) Val vs Test distribution shift

           metric            col  val_mean  test_mean   val_std  test_std   ks_stat      ks_p  wasserstein
        ATR ratio volatility_raw +0.009608  +0.006992 +0.005122 +0.002284 +0.327153 +0.000000    +0.002679
hourly log return     log_return +0.000014  +0.000029 +0.007114 +0.005213 +0.041502 +0.000000    +0.000970


## Figures

- ![menu_a](phase15_figures/menu_a_significance.csv)
- ![menu_b](phase15_figures/menu_b_distribution_shift.png)
- ![menu_c](phase15_figures/menu_c_three_env.png)