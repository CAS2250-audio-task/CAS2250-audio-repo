"""Functional checks for attention semantics, speech-local audio and resumable pruning."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from experiments.conformer.run import loaders, Clips
from experiments.conformer_sanity import speaker_fold
from experiments.conformer_long_zero.run import write_json
from experiments.conformer_aug.augment import draw_rng
from experiments.check_conformer_long_zero import recompute_predictions
from .data import Training
from .model import MaskedConformer, attention_bias, initialize
from .vad import frame_gates
from .waveforms import local_pitch, pitch_curve


def preflight(context, root, study, report):
    torch.set_num_threads(2)
    state, statistics = context["state"], context["statistics"]
    detector = context["detector"]
    wave = torch.from_numpy(sf.read(Path(context["cfg"].data_root) / "audio/출발/spk02_051.wav", dtype="float32")[0]).cuda()[None]
    p1 = detector.raw(wave)
    detector.raw(torch.randn_like(wave) * .03)
    p2 = detector.raw(torch.cat((wave, wave)))
    assert torch.max(torch.abs(p1 - p2[:1])) < .002
    assert torch.max(torch.abs(p2[:1] - p2[1:])) < 1e-6
    quiet, fallback = frame_gates(torch.zeros(2, 47, device="cuda"))
    assert quiet.eq(1).all() and fallback.all()
    model = MaskedConformer(state, statistics, "spk02", detector).cuda().eval()
    original = torch.load(root / "runs/overnight-v1/candidates/formant_B/spk02/best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(original["model"], strict=True)
    with torch.no_grad():
        base = model(wave)
        model.masking = True
        model.gate_override = lambda x: torch.ones(len(x), 38, device=x.device)
        unchanged = model(wave)
    assert torch.max(torch.abs(base - unchanged)) < 2e-5
    # Uniform attention logits make the expected suppression independently measurable.
    attention = model.transformer.encoder.layers[0].mha_layer
    zeros = torch.zeros(1, 4, 256, device="cuda")
    pos = torch.zeros(1, 7, 256, device="cuda")
    with torch.no_grad():
        _, weights = attention(zeros, zeros, zeros, pos_embs=pos)
        _, masked = attention(zeros, zeros, zeros, pos_embs=pos,
                              attn_mask=attention_bias(torch.tensor([[.05, 1., 1., .05]], device="cuda")))
    assert torch.allclose(weights, torch.full_like(weights, .25), atol=1e-6)
    assert masked[..., 0].max() < .025 and masked[..., 1].min() > .47
    model.gate_override = None
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(wave.repeat(2, 1))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 0], device="cuda")).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in detector.model.parameters())
    del model, original
    gc.collect()
    torch.cuda.empty_cache()
    # All variants must preserve background bit-for-bit and have actual turns in speech.
    t = np.arange(24000) / 16000
    synthetic = (np.sin(2 * np.pi * 170 * t) * .15).astype(np.float32)
    synthetic[:4000] = .01 * np.sin(2 * np.pi * 800 * t[:4000])
    regions = [(5000, 19000)]
    unique = set()
    for variant in range(8):
        changed, record = local_pitch(synthetic, "synthetic-word", variant, regions)
        curve, envelope, _ = pitch_curve("synthetic-word", variant, regions)
        assert np.array_equal(changed[envelope == 0], synthetic[envelope == 0])
        assert np.isfinite(changed).all() and changed.shape == synthetic.shape
        signs = np.sign(np.diff(curve[5000:19000]))
        assert (signs > 0).any() and (signs < 0).any()
        unique.add(changed.tobytes())
    assert len(unique) == 8
    import parselmouth
    from scipy.signal import lfilter
    vowel = sum(np.sin(2 * np.pi * 170 * k * np.arange(24000) / 16000) / k for k in range(1, 40))
    for frequency, bandwidth in ((600, 80), (1300, 100), (2500, 120)):
        radius = np.exp(-np.pi * bandwidth / 16000)
        vowel = lfilter([1 - radius], [1, -2 * radius * np.cos(2 * np.pi * frequency / 16000), radius ** 2], vowel)
    vowel = (vowel / np.abs(vowel).max() * .3).astype(np.float32)
    acoustic = []
    for variant in (0, 3, 7):
        altered, _ = local_pitch(vowel, "acoustic-check-vowel", variant, [(4000, 20000)])
        curve, _, _ = pitch_curve("acoustic-check-vowel", variant, [(4000, 20000)])
        pitch = parselmouth.Sound(altered, 16000).to_pitch_ac(time_step=.01, pitch_floor=100, pitch_ceiling=300)
        times, frequencies = pitch.xs(), pitch.selected_array["frequency"]
        valid = (times > .34) & (times < 1.16) & (frequencies > 0)
        measured = 12 * np.log2(frequencies[valid] / 170)
        requested = np.interp(times[valid] * 16000, np.arange(24000), curve)
        error = float(np.median(np.abs(measured - requested)))
        correlation = float(np.corrcoef(measured, requested)[0, 1])
        assert error < .15 and correlation > .95 and np.ptp(measured) > .4
        acoustic.append(dict(variant=variant, median_pitch_error_semitones=error, contour_correlation=correlation))
    write_json(report / "pitch_acoustic_check.json", dict(synthetic_f0_hz=170, checks=acoustic, all_passed=True))
    # Exact historical baseline replay and paired noise/source streams.
    replays = 0
    for speaker, bank in context["banks"].items():
        model = MaskedConformer(state, statistics, speaker, detector).cuda().eval()
        checkpoint = torch.load(root / f"runs/overnight-v1/candidates/formant_B/{speaker}/best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics, prediction, _ = bank.evaluate(model)
        expected = pd.read_csv(root / f"runs/overnight-v1/candidates/formant_B/{speaker}/best_off_predictions.csv.gz")
        joined = prediction.merge(expected, on="case_index", suffixes=("_new", "_old"), validate="one_to_one")
        assert joined.pred_idx_new.eq(joined.pred_idx_old).all()
        assert (joined.loss_new - joined.loss_old).abs().max() < 1e-6
        for key, value in recompute_predictions(prediction).items():
            assert abs(value - metrics[key]) < 1e-10
        replays += len(prediction)
        del model, checkpoint
        torch.cuda.empty_cache()
    frame = speaker_fold(context["metadata"], "spk02", context["cfg"])
    train_frame, val_frame = frame[frame.split.eq("train")], frame[frame.split.eq("val")]
    training = Training(train_frame, context["cfg"], "spk02", context["sources"], [], context["recordings"],
                        add_speech=False, focus_zero=False, formants=context["formants"],
                        pitch_bank=context["pitch_bank"], pitch=False)
    source = torch.load(root / "runs/conformer-aug-v1/folds/spk02/noise/last.pt", map_location="cpu", weights_only=True)
    train_loader, val_loader, _ = loaders(training, Clips(val_frame, context["cfg"]), 42, 16)
    train_loader.sampler.generator.set_state(source["loader_rng"])
    training.set_epoch(11)
    first = next(iter(train_loader))
    expected = pd.read_csv(root / "runs/overnight-v1/candidates/formant_B/spk02/epochs/augmentation_011.csv.gz").iloc[:16]
    actual = pd.DataFrame(training.trace)
    assert actual.path.tolist() == expected.path.tolist()
    assert actual.input_sha256.tolist() == expected.input_sha256.tolist()
    for blocks, count in ((8, 21567307), (6, 16289611)):
        model = MaskedConformer(state, statistics, "spk02", detector).cuda()
        optimizer = initialize(model, source, blocks=blocks)
        assert sum(p.numel() for p in model.parameters()) == count
        assert all(int(optimizer.state[p]["step"]) == 200 for group in optimizer.param_groups for p in group["params"])
        assert torch.equal(model.transformer.encoder.layers[-1].ffn_module1[1].ffn[0].weight.cpu(),
                           source["model"]["transformer.encoder.layers.11.ffn_module1.1.ffn.0.weight"])
        del model, optimizer
        torch.cuda.empty_cache()
    write_json(report / "preflight.json", dict(vad_batch_independent=True, empty_detection_safe=True,
        native_logits_preserved_with_unit_gate=True, masked_attention_suppression_verified=True,
        backward_finite=True, detector_frozen=True, eight_speech_local_curves_preserve_background=True,
        acoustic_pitch_contours_verified=True,
        historical_predictions_replayed=replays, historical_training_first16_exact=True,
        pruned_weights_and_adamw_states_preserved=True, test_audio_read=False, test_evaluated=False))
    # Benchmark a real optimizer step and canonical monitor with the new path active.
    results = []
    for masking in (False, True):
        model = MaskedConformer(state, statistics, "spk02", detector, masking=masking).cuda()
        optimizer = initialize(model, source)
        elapsed = []
        for _ in range(4):
            torch.cuda.synchronize()
            start = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            x, y = first[0].cuda(), first[1].cuda()
            detector.recent.clear()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed.append(time.monotonic() - start)
        start = time.monotonic()
        context["banks"]["spk02"].evaluate(model)
        evaluation = time.monotonic() - start
        seconds = float(np.median(elapsed[1:])) * 39 + evaluation
        results.append(dict(masking=masking, train_step_seconds=float(np.median(elapsed[1:])),
                            monitor_seconds=evaluation, estimated_epoch_seconds=seconds,
                            estimated_four_fold_90epoch_seconds=seconds * 360))
        del model, optimizer
        torch.cuda.empty_cache()
    write_json(report / "benchmark.json", dict(results=results, basis="39batch16 updates + canonical1827-case monitor; cache cleared for train VAD; excludes checkpoint/file overhead"))
