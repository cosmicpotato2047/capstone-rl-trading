# Phase 16d - pt OOS + DSR OOS mechanism analysis

## Menu 1: Test behavior per regime

variant vol_regime     n  trade_rate  hold_rate
   asym    low_vol 66070      0.0499     0.0320
   asym    mid_vol 68070      0.0517     0.0338
   asym   high_vol 66060      0.0517     0.0298
    dsr    low_vol 66070      0.0656     0.1308
    dsr    mid_vol 68070      0.0658     0.1217
    dsr   high_vol 66060      0.0617     0.1169
     pt    low_vol 66070      0.0421     0.0270
     pt    mid_vol 68070      0.0441     0.0286
     pt   high_vol 66060      0.0456     0.0261
    sym    low_vol 66070      0.0659     0.0631
    sym    mid_vol 68070      0.0662     0.0588
    sym   high_vol 66060      0.0626     0.0603


## Menu 2: Val vs Test behavior shift

variant  trade_rate_val  trade_rate_test  trade_rate_delta  hold_rate_val  hold_rate_test  hold_rate_delta  action_0_val  action_0_test  action_0_delta  action_1_val  action_1_test  action_1_delta
    sym         +0.0594          +0.0649           +0.0055        +0.0597         +0.0607          +0.0010       +0.1249        +0.1151         -0.0098       +0.0620        +0.0627         +0.0007
   asym         +0.0464          +0.0511           +0.0047        +0.0291         +0.0319          +0.0028       +0.3092        +0.2726         -0.0366       +0.0302        +0.0266         -0.0036
    dsr         +0.0589          +0.0644           +0.0054        +0.1274         +0.1231          -0.0043       +0.1242        +0.1187         -0.0055       +0.1508        +0.1462         -0.0047
     pt         +0.0392          +0.0440           +0.0048        +0.0247         +0.0272          +0.0025       +0.4294        +0.3788         -0.0506       +0.0538        +0.0440         -0.0097


## Menu 3: Hold duration on Test

variant  n_sessions  mean_duration_h  median_duration_h  p95_duration_h  max_duration_h
    sym        5666             2.15               1.00            5.00           98.00
   asym        4567             1.40               1.00            3.00            7.00
    dsr        5384             4.58               1.00           20.00          169.00
     pt        3922             1.39               1.00            3.00            6.00


## Figures

- ![Menu 1](phase16d_figures/menu1_test_behavior.png)
- ![Menu 2](phase16d_figures/menu2_val_test_shift.png)
- ![Menu 3](phase16d_figures/menu3_hold_duration.png)
- ![Menu 4](phase16d_figures/menu4_action_test.png)