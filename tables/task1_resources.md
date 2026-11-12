| Run | trainable params (M) | trainable (%) | train time (min) | setup (s) | checkpoint writes (s, total) | peak GPU mem (GB) | weights file (MB) | optimizer file (MB) | OPLoRA projected export (MB) | checkpoints saved |
|---|---|---|---|---|---|---|---|---|---|---|
| Full FT | 596.05 | 100.0 | 76.0 | 0.52 | 29.58 | 13.23 | 2273.87 | 4547.89 | nan | 4.0 |
| Full FT + L2-SP | 596.05 | 100.0 | 76.73 | 0.48 | 29.12 | 15.46 | 2273.87 | 4547.94 | nan | 4.0 |
| LoRA-4 | 2.52 | 0.42 | 109.57 | 0.41 | 0.45 | 4.7 | 9.75 | 19.7 | nan | 4.0 |
| LoRA-16 | 10.09 | 1.67 | 109.03 | 0.51 | 0.8 | 4.81 | 38.62 | 77.45 | nan | 4.0 |
| OPLoRA-16 | 10.09 | 1.67 | 162.59 | 15.36 | 1.18 | 5.11 | 38.62 | 77.45 | 38.63 | 4.0 |
