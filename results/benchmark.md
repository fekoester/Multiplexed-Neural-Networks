| task | model | main | control | parameters | test accuracy (mean ± std, 3 seeds) | best epoch acc | s / run |
|---|---|---|---|---|---|---|---|
| gated | MLP | 16-64-4 | – | 1348 | 83.77 ± 0.41 % | 84.05 % | 1.4 |
| gated | MLP (wide) | 16-503-4 | – | 10567 | 83.62 ± 0.44 % | 83.62 % | 9.0 |
| gated | MLP (deep) | 16-92-92-4 | – | 10492 | 90.82 ± 0.19 % | 91.03 % | 7.0 |
| gated | multiplexed (no hidden layer) | 16-4 | 16 | 1156 | 95.75 ± 0.13 % | 95.86 % | 0.6 |
| gated | multiplexed | 16-32-4 | 16-8 | 10572 | 93.80 ± 0.28 % | 94.57 % | 4.9 |
| letter | MLP | 16-32-26 | – | 1402 | 86.10 ± 0.30 % | 86.19 % | 2.0 |
| letter | MLP (wide) | 16-397-26 | – | 17097 | 95.21 ± 0.18 % | 95.32 % | 22.2 |
| letter | MLP (deep) | 16-128-96-26 | – | 17082 | 95.69 ± 0.14 % | 95.92 % | 19.2 |
| letter | multiplexed (first layer) | 16-32-26 | 16 | 10106 | 94.38 ± 0.33 % | 94.45 % | 9.7 |
| letter | multiplexed | 16-32-26 | 16-8 | 17106 | 95.95 ± 0.24 % | 96.27 % | 15.9 |
