# Final paired boundary detector results

Both 300-epoch runs completed in 43.81 minutes. Validation: S02 노정민, 200 originals; each mixed at nominal 0dB with eight held-out noise recordings (1,600 mixtures), plus 80 noise-only windows. S01 heldout not evaluated. Results use minimum asymmetric-duration-cost checkpoint, no crop margin. Event F1 uses IoU >= 0.5, not command accuracy or precision of endpoints.

| Metric | TCN | CNN-GRU |
|---|---:|---:|
| Selected epoch | 60 | 109 |
| Original event F1 | 73.60% | 95.88% |
| Noisy event F1 | 58.89% | 70.71% |
| Original matched start/end MAE | 20.5 / 98.5 ms | 9.5 / 34.3 ms |
| Noisy matched start/end MAE | 60.3 / 82.3 ms | 77.9 / 59.2 ms |
| Noisy multiple-interval outputs | 707 / 1600 | 425 / 1600 |
| Noise-only windows with any predicted speech | 35 / 80 | 30 / 80 |
| Weighted duration cost per validation input | 572.6 ms | 524.4 ms |

Endpoint MAE excludes unmatched cases; it must not be interpreted as whole-dataset crop accuracy. The weighted duration cost is also not timestamp MAE. A noisy input with exactly one predicted interval and BOTH endpoints within +/-30ms of truth: TCN127/1600 (7.94%), GRU113/1600 (7.06%). This deliberately strict practical criterion penalizes extra detections and boundary mismatch together. It shows that event F1 substantially overstates readiness for exact automatic cropping.

## Noise-specific event F1

| Noise | TCN | CNN-GRU |
|---|---:|---:|
| Cafeteria | 29.33% | 49.11% |
| Car | 94.96% | 98.76% |
| Compressor | 52.51% | 40.16% |
| Fan | 62.13% | 83.53% |
| Restaurant | 31.87% | 44.89% |
| Station | 78.10% | 85.11% |
| Traffic | 86.57% | 92.91% |
| Washing | 89.84% | 97.06% |

The cafeteria/restaurant labels suggest backgrounds may include other speech, but no listening audit has established that for these exact crops. A generic speech detector and a target-command speaker detector have different labels if background talk is present. Audit before attributing these errors to model size.

## What failed and what remains uncertain

- Noise-specific generalization and extraneous events are measured failures. Existing six train recordings supply many windows but few independent environments. More epochs do not add new environments.
- Leading speech is often cut. On matched noisy events, TCN1092/1315 and GRU1170/1311 start >10ms late. Ending cuts are TCN395/1315 and GRU414/1311. GRU has smaller average absolute ending error but does not have fewer >10ms end cuts at its selected checkpoint. We did not prove that the asymmetric training objective improves ending cuts: there is no symmetric-loss control.
- Causal models can plausibly recognize weak onsets only after subsequent evidence. This is an architectural hypothesis; probabilities and aligned examples need inspection. Centered frontend also has lookahead, so implementation is not strictly zero-lookahead.
- Training adds weight only within 50ms of true boundaries; evaluation weights ALL leading/trailing missing duration. Framewise BCE is a surrogate, not explicit start/end optimization. Raising the end weight alone cannot guarantee accurate boundaries.
- Global selection trades background inclusion and truncation across conditions. TCN's maximum original F1 was93.87% at142, not its selected60; maximum noisy F1 was62.57% at191. GRU maxima: original97.55% at149, noisy71.51% at136. These are separate checkpoints and cannot be combined into one model result. The particularly weak original TCN number partly reflects the global selection rule.
- Both models' best weighted costs precede epoch300; train loss continues down while final validation cost rises (TCN572.6→642.3, GRU524.4→589.8ms). This is consistent with overfitting or worsening generalization; extending to1000 is not supported.

## Margins and next diagnostics

Validation margin sweep selected TCN before30/after20ms (cost531.0ms), GRU before30/after30ms (458.6ms). Recommendations are not applied automatically. For GRU symmetric30ms, >10ms matched noisy end cuts decrease414→261, but strict single-interval +/-30ms crop success is still only128/1600 (8.00%). Fixed margins cannot fix severe fragmentation or environment errors; expanding can worsen excess background and some exact endpoint errors.

Before more training, inspect waveforms, probability tracks and labels for onset-delayed, fragmented and compressor cases, and listen to background-talk examples. Evaluate decoding gap-merging/minimum-duration options on validation with explicit noise-only false alarms; preserve raw decoder baseline. Then consider an offline local-context boundary model and more independent noise recordings. No additional training or holdout tuning launched during this review.

Detailed numeric breakdown: paired_300_results.json. Checkpoints and raw best predictions remain under runs/boundary-300-v1/{tcn,cnn_gru}/.
