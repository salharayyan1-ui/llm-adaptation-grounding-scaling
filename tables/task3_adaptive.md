| Dataset | Strategy | policy | accuracy % [95% CI] | mean tokens | mean rollouts | stopped at 4/8/16/32 (%) | early incorrect stops recovered by fixed-32 |
|---|---|---|---|---|---|---|---|
| math | adaptive: agreement | agreement(tau=0.5,gamma=0.0) | 72.2 [68.9, 75.1] | 6790 | 8.0 | 82 / 4 / 1 / 14 | 183/1415 |
| math | adaptive: quality-aware | quality(tau=0.5,gamma=0.0) | 72.7 [69.5, 75.6] | 7645 | 9.0 | 77 / 5 / 1 / 16 | 129/1212 |
| math | fixed SC N=4 | fixed N=4 | 68.3 [65.2, 71.1] | 2669 | 4 | — | — |
| math | fixed SC N=8 | fixed N=8 | 71.2 [68.1, 74.0] | 5344 | 8 | — | — |
| math | fixed SC N=16 | fixed N=16 | 72.8 [69.6, 75.8] | 10680 | 16 | — | — |
| math | fixed SC N=32 | fixed N=32 | 73.4 [70.1, 76.4] | 21347 | 32 | — | — |
| mathvista | adaptive: agreement | agreement(tau=0.5,gamma=0.0) | 63.0 [60.3, 65.9] | 2908 | 8.4 | 77 / 7 / 2 / 14 | 309/2054 |
| mathvista | adaptive: quality-aware | quality(tau=0.5,gamma=0.0) | 63.4 [60.7, 66.4] | 3324 | 9.7 | 70 / 9 / 3 / 18 | 246/1825 |
| mathvista | fixed SC N=4 | fixed N=4 | 59.4 [56.8, 62.1] | 1118 | 4 | — | — |
| mathvista | fixed SC N=8 | fixed N=8 | 61.3 [58.5, 64.2] | 2226 | 8 | — | — |
| mathvista | fixed SC N=16 | fixed N=16 | 62.9 [60.1, 65.9] | 4451 | 16 | — | — |
| mathvista | fixed SC N=32 | fixed N=32 | 63.9 [60.7, 67.1] | 8898 | 32 | — | — |
