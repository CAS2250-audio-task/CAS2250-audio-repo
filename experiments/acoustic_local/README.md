# local + legacy acoustic set3

Base: origin/compact-pooling-v1 @b00956b, containing merged local-route models and the separate frequency-axis ablations. Branch: experiment/compact-local-acoustic-aux. This experiment uses basic local CNN, with no added vertical CNN/attention.

All three readouts (Mean/Pooled/Self) gain an auxiliary input route. Legacy set3 is F1/F2/F3/B1/B2/B3 plus six validity channels; this is feature fusion, not an additional target/loss. Missing measurements are zero after normalization with their availability0. Availability is not a quality guarantee. Actual noisy waveform is used to extract noisy-view features. Training-reference auxiliaries are never substituted for noisy-view features.

A temporal CNN12→16→32→32,k5/k5/k1,GroupNorm/GELU maps151 frames to151 frames and adds a32D residual before the readout. Final projection is zero initialized: all starting logits equal the existing local model. Both CNN paths train; the shared architecture/initial weights and auxiliary weights are matched across independent models. Scaler is fit only from609 original training recordings' observed frames. Scaler buffers are saved in each model state, and the scaler is also in the fixed protocol.

Extract only legacy set3 to avoid unnecessary pYIN computation. Burg settings and digitally-empty-window support match experiments/overnight/features.py. Extractor returns raw6+availability6. Backbone normalizes using saved scaler then fuses. Inputs to network are `[B,76,151]`, first64 normalized logmel bands and final12 raw auxiliary/availability. Inference requires both paths and includes descriptor extraction cost; do not quote only neural-network latency as total latency.

Protocol: spk03 validation203,train spk02/04/05=609,seed42,1000epochs from scratch,AdamW1e-3→1e-5cosine,batch32 original/64paired views,WD1e-4,CE+0.1SupCon same_command,temp0.1. Same formant/noise seeds and cached input sources as local reference. No test01 audio. Existing local results/checkpoints reused as controls. No LR/batch search. Best0dB draw0 checkpoint evaluated at8noise×3draw plus original and noise-only. Also inspect accuracy and command-only F1 to separate silence effects.

```bash
python -m experiments.acoustic_local.check
python -m experiments.acoustic_local.run --aux-workers 4 \
  --objective paired_supcon --positive-mode same_command \
  --supcon-weight 0.1 --temperature 0.1 --folds spk03 \
  --epochs 1000 --output runs/compact-acoustic-local-spk03-e1000
```

Dependencies are the compact trainer stack plus praat-parselmouth0.4.7. The check's legacy librosa0.8 comparison uses NumPy compatibility aliases only in the check process. Training extractor uses NumPy and parselmouth directly. `launch_acoustic_local.py` provides detached supervision and auto-reporting. This branch pins only spk03 to prevent reusing its scaler on a different fold. Train cache/noise paths must be restored to this checkout first.
