# Boundary detector paired experiment, 2026-11-16

Branch: experiment/speech-boundary-detector. User selected TCN + CNN-GRU, 300 epochs each.

## Data and split

Use full trimmed_originals from manual-clipped-v1, with speech_start_in_trimmed_s / speech_end_in_trimmed_s. Do not use clip output bounds (up to 30ms margins) as speech labels.

- Train: S03 곽민채, S04 최환준, S05 이서준, 600 originals.
- Validation: S02 노정민 (old dataset spk03), 200 originals.
- Holdout: S01 강재훈, 200 originals, excluded from training and validation.
- Existing 6 training noise recordings, original source groups disjoint from 8 validation noise recordings. No new noise dataset added in this baseline.
- Training SNR uniform -5 to 20 dB, 80% mixing probability for positive windows; mix entire original using annotated speech-region power. Four sampled windows per source per epoch, including one real external-noise-only window. 1.5s windows with padding excluded from loss.
- Validation: each original alone and mixed with each of 8 reserved noises at fixed nominal 0 dB, plus 10 fixed noise-only windows per validation source. Fixed mixtures cached as features, shared recipe and seed across models. The source is not necessarily denoised; SNR refers to additional noise.

## Models and optimization

64 log-mel bins, 25ms analysis window, 5ms hop. TCN width64: 22,465 parameters. CNN-GRU width64, unidirectional hidden32: 18,209 parameters. Channel normalization, causal temporal layers; centered acoustic frontend has lookahead, so this is not a complete streaming implementation.

Both: AdamW, LR 0.001 cosine to 0.00001 over 300 epochs, weight decay 0.0001, batch32, seed42, gradient norm clip5. Mean/std estimated only on original training data. Independent model weights, same data protocol.

## Asymmetric objective

Labels stay at actual annotated speech boundaries. In training BCE, positive frames within 50ms of onset carry weight3; those within 50ms of offset carry weight5; other frames have weight1. If the two bands overlap, offset weight wins. This is a framewise surrogate, not a differentiable endpoint regression loss. It encourages retaining weak boundary speech but can include more background.

Best checkpoint minimizes validation duration cost: leading missing speech x3, trailing missing speech x5, internal missed speech x5, complete missed speech x5, excess included background x1. Includes all gold and predicted intervals, even failed IoU matches and empty predictions. Reports ordinary BCE, frame/event F1, matched endpoint MAE/p95, truncation counts, misses and noise-only false events separately. Cost in ms is weighted duration per validation case, not literal timestamp MAE. Weights are an initial user-directed setting, not tuned values.

Decode on/off thresholds .6/.4, confirmation20/50ms with backdating; no default crop margin. After training, sweep onset/offset margins separately over 0/10/20/30ms using best-checkpoint validation predictions only; save recommended setting and full grid. No holdout tuning. Margin sweep recommendations require explicit application at deployment; predict CLI's existing --margin-ms applies the same margin on both sides.

## Running and checking

Python: /root/CAS2250-resnet18-pretrain-100/.venv/bin/python

Supervisor: `python -m experiments.boundary_detector.run_pair`

Artifacts: runs/boundary-300-v1/{tcn,cnn_gru}/ with config, source/data hashes, history, progress, last/best checkpoints, best_predictions.json, margin_sweep.json. Both runs execute sequentially. Re-running supervisor resumes identical-config checkpoints and stops on child errors. pair_progress.json and per-model logs expose failures. Shell-independent detached process, UTF-8 enabled.

Checks: five unit checks passed: TCN and GRU shape/gradient/causality, decoding backdating, mapping/padding/split filtering, speech-region SNR, asymmetric miss/extra/end costs. Synthetic two-epoch training, resume and inference passed with cached validation and weighted BCE before paired launch.

Limitations: one validation speaker, one seed, limited six training noise environments, manual boundary labels not independently audited by listening. Boundary improvements require downstream command-recognition evaluation after training. This is a boundary detector, not a command classifier.

Launch verification: TCN passed epoch16/300 with finite loss and metrics; measured ~4.4 seconds/epoch, ~21 minutes remaining at epoch16. CNN-GRU completed an independent 2-epoch CUDA synthetic smoke check. Paired completion estimate 45–60 minutes from launch; CNN-GRU ETA remains approximate until its own real epochs run.
