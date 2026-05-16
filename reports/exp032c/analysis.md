# exp032c Mechanism Analysis

Source: `exp032c_trajectories.parquet` (1,042,960 step rows, 40 models)


## Key findings

- **Menu 1 Pareto frontier**: 5/40 runs lie on Pareto frontier in Sharpe-MDD plane.
- **Menu 5 Policy distance**: within-cluster L2 = 0.129, across-cluster L2 = 0.286 (ratio 2.22x). → Cluster separation statistically confirmed at policy level.

## Menu 4 trade rate per regime (high_vol)

- asym: 0.0377 (~330 trades/year if persistent)
- dsr: 0.0468 (~410 trades/year if persistent)
- pt: 0.0318 (~279 trades/year if persistent)
- sym: 0.0471 (~413 trades/year if persistent)

## Figures

- ![menu1_pareto_scatter.png](exp032c_figures/menu1_pareto_scatter.png)
- ![menu2_action_distribution.png](exp032c_figures/menu2_action_distribution.png)
- ![menu3_counterfactual_action_map.png](exp032c_figures/menu3_counterfactual_action_map.png)
- ![menu4_behavior_per_regime.png](exp032c_figures/menu4_behavior_per_regime.png)
- ![menu5_policy_distance.png](exp032c_figures/menu5_policy_distance.png)