# Detector diversity v2 results

Selected by the same legacy validation asymmetric duration cost; no crop margins. New environmental sources evaluated only after checkpoint selection.

| Model | Params | Best epoch | Original event F1 | Legacy0dB event F1 | Noisy speech missing | Extra ms/case | Noise-only FP files |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 18209 | 163 | 96.60% | 75.59% | 21.88% | 120.4 | 32.5% |
| lookahead | 18209 | 152 | 75.52% | 69.62% | 13.23% | 123.1 | 77.5% |
| boundary | 18275 | 210 | 97.31% | 76.34% | 20.00% | 134.1 | 36.2% |
| spectral | 23593 | 183 | 98.04% | 73.12% | 20.03% | 167.7 | 36.2% |

## Fixed-checkpoint noise severity and new-environment audit

| Model | Suite | Event F1 (noisy) | Speech missing (noisy) | Extra ms/case (noisy) |
|---|---|---:|---:|---:|
| baseline | legacy_0db | 75.59% | 21.88% | 120.4 |
| lookahead | legacy_0db | 69.62% | 13.23% | 123.1 |
| boundary | legacy_0db | 76.34% | 20.00% | 134.1 |
| spectral | legacy_0db | 73.12% | 20.03% | 167.7 |
| v1_legacy_noise | legacy_0db | 70.71% | 19.84% | 201.3 |
| baseline | legacy_5db | 86.28% | 14.20% | 102.0 |
| lookahead | legacy_5db | 74.16% | 7.28% | 105.9 |
| boundary | legacy_5db | 85.07% | 13.03% | 109.4 |
| spectral | legacy_5db | 82.00% | 13.92% | 140.2 |
| v1_legacy_noise | legacy_5db | 78.46% | 12.35% | 170.3 |
| baseline | legacy_10db | 89.71% | 11.22% | 83.9 |
| lookahead | legacy_10db | 75.11% | 5.60% | 90.9 |
| boundary | legacy_10db | 88.55% | 10.46% | 89.7 |
| spectral | legacy_10db | 86.00% | 11.76% | 110.4 |
| v1_legacy_noise | legacy_10db | 82.72% | 9.33% | 136.3 |
| baseline | new_environment_0db | 72.22% | 23.27% | 123.2 |
| lookahead | new_environment_0db | 62.27% | 17.92% | 131.4 |
| boundary | new_environment_0db | 72.81% | 21.89% | 129.4 |
| spectral | new_environment_0db | 71.96% | 21.37% | 99.4 |
| v1_legacy_noise | new_environment_0db | 47.15% | 22.61% | 447.7 |

## GPU latency

| Model | Median ms | p95 ms | Added lookahead ms |
|---|---:|---:|---:|
| baseline | 1.054 | 1.087 | 0.0 |
| lookahead | 1.068 | 1.086 | 80.0 |
| boundary | 1.207 | 1.234 | 0.0 |
| spectral | 1.283 | 1.302 | 0.0 |
| v1_legacy_noise | 1.036 | 1.055 | 0.0 |

Frontend adds nominal12.5ms analysis lookahead to all models. GPU timings include mel+model on1.5s audio, batch1, excluding IO; not per-frame streaming timing.

Speech missing counts only uncovered gold time; extra/background time is reported separately. Event F1 and speech retention can rank models differently. Thresholds.6/.4, confirmation20/50ms backdated are held fixed. Boundary heads are training auxiliary heads, not a changed decoder.

Limits: one validation speaker/one primary seed; original may already contain noise; categorical noise selection is not a listening audit; no intentional speech/babble augmentation. New22sources are disjoint original recordings but largely same broad categories, not necessarily novel semantic noise classes. No S01 speech holdout evaluation.
