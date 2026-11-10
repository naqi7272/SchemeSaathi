# Results

Mean with 95% cluster-bootstrap CI. Percent unless noted. n/a = not applicable.
Code-scored metrics come first; they need no judge.

- `accuracy` (LLM judge) not reported: no judge validation (results\judge_validation.json missing).
- `faithful` (LLM judge) not reported: no judge validation (results\judge_validation.json missing).

## numbers_exact (%)

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | 2.4 [0.0, 8.1] | 4.5 [0.0, 13.6] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| arm2-rag-base-q8 | 52.4 [36.2, 68.9] | 63.6 [40.9, 81.8] | 0.0 [0.0, 0.0] | 72.7 [45.5, 100.0] |
| arm2b-rag-translate-base-q8 | 52.4 [32.7, 72.5] | 63.6 [40.9, 81.8] | 22.2 [0.0, 55.6] | 54.5 [27.3, 81.8] |

## refused_unanswerable (%)

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | 71.4 [42.9, 100.0] | 75.0 [25.0, 100.0] | 50.0 [0.0, 100.0] | 100.0 [100.0, 100.0] |
| arm2-rag-base-q8 | 42.9 [0.0, 77.8] | 50.0 [0.0, 100.0] | 50.0 [0.0, 100.0] | 0.0 [0.0, 0.0] |
| arm2b-rag-translate-base-q8 | 42.9 [0.0, 77.8] | 50.0 [0.0, 100.0] | 50.0 [0.0, 100.0] | 0.0 [0.0, 0.0] |

## false_refusal (%)

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | 53.7 [38.8, 67.2] | 36.7 [20.0, 53.3] | 72.7 [45.5, 100.0] | 76.9 [53.8, 100.0] |
| arm2-rag-base-q8 | 3.7 [0.0, 8.8] | 0.0 [0.0, 0.0] | 18.2 [0.0, 45.5] | 0.0 [0.0, 0.0] |
| arm2b-rag-translate-base-q8 | 3.7 [0.0, 9.1] | 0.0 [0.0, 0.0] | 18.2 [0.0, 45.5] | 0.0 [0.0, 0.0] |

## citation_precision (%)

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | n/a | n/a | n/a | n/a |
| arm2-rag-base-q8 | 23.1 [11.7, 36.1] | 28.3 [15.0, 43.3] | 0.0 [0.0, 0.0] | 26.9 [7.7, 50.0] |
| arm2b-rag-translate-base-q8 | 24.5 [13.0, 38.0] | 28.3 [15.0, 43.3] | 22.2 [0.0, 55.6] | 16.7 [0.0, 41.7] |

## citation_recall (%)

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| arm2-rag-base-q8 | 25.0 [13.0, 38.3] | 31.7 [16.7, 48.3] | 0.0 [0.0, 0.0] | 26.9 [7.7, 50.0] |
| arm2b-rag-translate-base-q8 | 25.0 [13.0, 38.6] | 31.7 [16.7, 48.3] | 22.2 [0.0, 55.6] | 11.5 [0.0, 30.8] |

## citations_grounded (%)

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | n/a | n/a | n/a | n/a |
| arm2-rag-base-q8 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| arm2b-rag-translate-base-q8 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |

## retrieval_recall (%)

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | n/a | n/a | n/a | n/a |
| arm2-rag-base-q8 | 33.3 [20.8, 46.9] | 41.7 [25.0, 60.0] | 0.0 [0.0, 0.0] | 42.3 [15.4, 69.2] |
| arm2b-rag-translate-base-q8 | 35.2 [21.4, 50.0] | 41.7 [25.0, 60.0] | 18.2 [0.0, 45.5] | 34.6 [11.5, 61.5] |

## number_recall (%)

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | 11.8 [2.2, 24.1] | 16.7 [2.8, 34.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| arm2-rag-base-q8 | 61.3 [46.3, 75.9] | 70.5 [52.3, 86.4] | 7.1 [0.0, 21.4] | 77.3 [53.0, 100.0] |
| arm2b-rag-translate-base-q8 | 66.7 [50.5, 82.2] | 70.5 [52.3, 86.4] | 56.0 [28.6, 82.1] | 65.9 [42.4, 87.9] |

## script_match (%)

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | 95.1 [89.6, 100.0] | 100.0 [100.0, 100.0] | 76.9 [53.8, 100.0] | 100.0 [100.0, 100.0] |
| arm2-rag-base-q8 | 95.1 [89.8, 100.0] | 100.0 [100.0, 100.0] | 76.9 [53.8, 100.0] | 100.0 [100.0, 100.0] |
| arm2b-rag-translate-base-q8 | 80.3 [71.8, 89.2] | 100.0 [100.0, 100.0] | 15.4 [0.0, 38.5] | 92.9 [78.6, 100.0] |

## parse_error (%)

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | 1.6 [0.0, 5.1] | 0.0 [0.0, 0.0] | 7.7 [0.0, 23.1] | 0.0 [0.0, 0.0] |
| arm2-rag-base-q8 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| arm2b-rag-translate-base-q8 | 1.6 [0.0, 5.4] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 7.1 [0.0, 21.4] |

## latency_s (s)

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | 17.91 [13.77, 22.98] | 15.03 [12.56, 18.81] | 27.52 [17.88, 44.02] | 15.99 [10.01, 25.51] |
| arm2-rag-base-q8 | 115.06 [94.79, 134.78] | 70.78 [62.65, 79.78] | 280.57 [246.58, 325.08] | 68.93 [58.32, 80.99] |
| arm2b-rag-translate-base-q8 | 126.26 [104.01, 153.18] | 123.34 [92.47, 163.12] | 123.67 [97.42, 163.52] | 135.73 [96.41, 177.64] |

## output_tokens

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | 62.48 [51.07, 78.63] | 53.47 [48.47, 58.59] | 105.00 [62.23, 178.15] | 44.86 [34.50, 57.36] |
| arm2-rag-base-q8 | 77.61 [65.78, 92.83] | 67.68 [61.06, 74.44] | 111.08 [70.00, 171.39] | 70.64 [56.86, 88.79] |
| arm2b-rag-translate-base-q8 | 85.64 [72.78, 102.60] | 67.68 [61.06, 74.44] | 101.77 [83.38, 122.85] | 114.29 [73.36, 183.21] |

## cost_usd

| run | all | en | hi | hinglish |
|---|---|---|---|---|
| arm1-prompt-base-q8 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| arm2-rag-base-q8 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| arm2b-rag-translate-base-q8 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

## Paired comparisons

diff = A - B (percentage points). p_holm: cluster-bootstrap p, Holm-adjusted across all rows. McNemar p is unadjusted, for reference.

| A | B | metric | language | n | diff [95% CI] | p_holm | McNemar p |
|---|---|---|---|---|---|---|---|
| arm2-rag-base-q8 | arm1-prompt-base-q8 | numbers_exact | all | 42 | +50.0 [+33.3, +65.9] | 0 | 9.54e-07 |
| arm2-rag-base-q8 | arm1-prompt-base-q8 | numbers_exact | en | 22 | +59.1 [+36.4, +77.3] | 0 | 0.000244 |
| arm2-rag-base-q8 | arm1-prompt-base-q8 | numbers_exact | hi | 9 | +0.0 [+0.0, +0.0] | 1 | 1 |
| arm2-rag-base-q8 | arm1-prompt-base-q8 | numbers_exact | hinglish | 11 | +72.7 [+45.5, +100.0] | 0 | 0.00781 |
| arm2-rag-base-q8 | arm1-prompt-base-q8 | refusal_ok | all | 61 | +41.0 [+24.5, +55.7] | 0 | 4.65e-06 |
| arm2-rag-base-q8 | arm1-prompt-base-q8 | refusal_ok | en | 34 | +29.4 [+8.8, +47.1] | 0.03 | 0.0129 |
| arm2-rag-base-q8 | arm1-prompt-base-q8 | refusal_ok | hi | 13 | +46.2 [+23.1, +69.2] | 0.0096 | 0.0312 |
| arm2-rag-base-q8 | arm1-prompt-base-q8 | refusal_ok | hinglish | 14 | +64.3 [+28.6, +92.9] | 0.0154 | 0.0117 |
| arm2b-rag-translate-base-q8 | arm2-rag-base-q8 | numbers_exact | all | 42 | +0.0 [-9.1, +9.3] | 1 | 1 |
| arm2b-rag-translate-base-q8 | arm2-rag-base-q8 | numbers_exact | en | 22 | +0.0 [+0.0, +0.0] | 1 | 1 |
| arm2b-rag-translate-base-q8 | arm2-rag-base-q8 | numbers_exact | hi | 9 | +22.2 [+0.0, +55.6] | 1 | 0.5 |
| arm2b-rag-translate-base-q8 | arm2-rag-base-q8 | numbers_exact | hinglish | 11 | -18.2 [-45.5, +0.0] | 1 | 0.5 |
| arm2b-rag-translate-base-q8 | arm2-rag-base-q8 | refusal_ok | all | 61 | +0.0 [-6.7, +6.3] | 1 | 1 |
| arm2b-rag-translate-base-q8 | arm2-rag-base-q8 | refusal_ok | en | 34 | +0.0 [+0.0, +0.0] | 1 | 1 |
| arm2b-rag-translate-base-q8 | arm2-rag-base-q8 | refusal_ok | hi | 13 | +0.0 [-30.8, +30.8] | 1 | 1 |
| arm2b-rag-translate-base-q8 | arm2-rag-base-q8 | refusal_ok | hinglish | 14 | +0.0 [+0.0, +0.0] | 1 | 1 |
