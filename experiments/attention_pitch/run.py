"""Detached paired4fold study; mandatory core first, time-gated depth reduction last."""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

from src import engine
from src.config import load_config
from src.dataset import read_metadata
from experiments.conformer.model import load_source
from experiments.conformer.run import Clips, loaders, atomic_save
from experiments.conformer_sanity import SPEAKERS, speaker_fold
from experiments.conformer_long_zero.run import ValidationBank, capture, write_json
from experiments.conformer_long_zero.policy import update_count
from experiments.check_conformer_long_zero import recompute_predictions
from .data import LegacyFormants, Training
from .model import MaskedConformer, initialize
from .vad import Detector, wave_hash
from .waveforms import prepare
from .report import render


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/attention-pitch-v1"
REPORT = ROOT / "reports/attention-pitch-v1"
SPECS = [dict(name="attention", masking=True, pitch=False, blocks=12),
         dict(name="pitch", masking=False, pitch=True, blocks=12),
         dict(name="attention_pitch", masking=True, pitch=True, blocks=12)]
BASE = dict(name="baseline", masking=False, pitch=False, blocks=12)


def progress(**values):
    values.update(pid=os.getpid(), updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json(STUDY / "progress.json", values)
    write_json(REPORT / "progress.json", values)
    print("PROGRESS", json.dumps(values), flush=True)


def freeze():
    parent = json.loads((ROOT / "reports/overnight-v1/protocol.json").read_text(encoding="utf-8"))
    pins = dict(parent["source_sha256"])
    for path, digest in pins.items():
        assert engine.file_hash(ROOT / path) == digest, path
    pins.update({str(p.relative_to(ROOT)): engine.file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    vad = json.loads((STUDY / "assets/silero_source.json").read_text(encoding="utf-8"))
    assert engine.file_hash(STUDY / "assets/silero_vad.jit") == vad["sha256"]
    assets = {str(p.relative_to(ROOT)): engine.file_hash(p) for p in (STUDY / "assets").glob("*")}
    banks = {p: digest for p, digest in parent["validation_bank_sha256"].items() if "/filtered/" not in p}
    for path, digest in banks.items():
        assert engine.file_hash(ROOT / path) == digest, path
    for path, digest in parent["development_wav_sha256"].items():
        assert "spk01_" not in path
        assert engine.file_hash(Path(parent["data_root"]) / path) == digest
    for path, digest in parent["initial_checkpoint_sha256"].items():
        assert engine.file_hash(ROOT / path) == digest
    value = dict(study="attention-pitch-v1", seed=42, speakers=list(SPEAKERS), batch=16,
        epochs=100, continuation_epochs=90, draws_per_epoch=609, early_stopping=False,
        encoder_lr=1e-5, head_lr=5e-4, weight_decay=1e-4, encoder_dropout=.2,
        frontend_dropout=.1, no_freezing=True, no_lr_search=True, auxiliary_set="0", denoiser="off",
        initial_state="Exact baseline-noise epoch10 LAST model, AdamW states, sampler and Torch/CUDA RNG; all candidates11..100; baseline reused after exact replay",
        baseline=BASE, core=SPECS, common_noise="p0.5, uniform[-5,20]dB, existing6 sources/crop RNG",
        common_formants="Original selected policy p0.5,2variants, loguniform[1/1.1,1.1]; existing cached formant-only waves, unchanged",
        waveform_order="new speech-local pitch -> unchanged common formant -> online noise; no speed/SpecAugment",
        pitch=dict(probability=.5,variants=8,curve="PCHIP with3/4 random interior knots alternating direction, zero endpoints,8deterministic variants per file",
                   method="Praat ToManipulation10ms65..600Hz, multiply original pitch-tier frequencies, overlap-add resynthesis; no voiced points -> unchanged waveform",
                   maximum_semitones=[1.,1.5,2.,2.5],maximum_slope_st_per_second=32,
                   training_boundary_source="Existing automatic Silero VAD segments mapped to fixed1.5s; crop intersection; not manually certified ground truth",
                   outside_speech="Bitwise original before legacy common formant transformation;20ms crossfade inside boundary; added noise always after transformations",
                   overlap_limitation="Pre-existing background overlapping speech cannot be independently preserved without source separation"),
        vad=dict(**vad,window_samples=512,windows_per_clip=47,reset_states_each_independent_batch=True,
                 input="Only actual augmented/mixed waveform, never original-side boundaries for classifier inference",
                 alignment="VAD window centers -> CNN centers every640samples",smooth_frames=3,boundary_margin_ms=80,
                 gate="g=.05+.95*p; if max expanded p<.3 use all-one gates",attention="add log(g_j) to each head's key logits in all blocks",
                 pooling="sum(g_t*h_t)/sum(g_t)",conv_module="unchanged; indirect background influence remains"),
        checkpoint_selection="Same fixed0dB draw0 mean of8per-noise macroF1, earliest strict maximum, include epoch10",
        final_evaluation="Same all3draws0dB+original+noise-only, canonical batch32 per203/20unit, no test01",
        condition_gate="MeanF1 gain>=.005 and no individual speaker decline>.01 vs baseline; report raw best separately",
        compression=dict(blocks=[8,6],selection="equally spaced original layers including0and11, nearest integer",
                         init="Same epoch10 source and retained parameters' exact AdamW moments; no extra training budget or distillation",
                         eligibility="vs chosen12block: meanF1 loss<=.005, eachspeaker loss<=.01, meanoriginal accuracy loss<=.005",
                         budget="optional only, require max(actual core condition time,45minutes)+20minutes reserve before7hour target; never truncate a core fold"),
        diagnostics="Existing source-VAD boundary priors and pool-only effects at unchanged baseline best checkpoints; informative original-side input, never used to select deployable model",
        retention="Keep all best models, histories, epoch predictions and draw traces; remove only new completed LAST optimizer after fold audit",
        source_sha256=pins,asset_sha256=assets,validation_bank_sha256=banks,
        initial_checkpoint_sha256=parent["initial_checkpoint_sha256"],development_wav_sha256=parent["development_wav_sha256"],
        parent_protocol_sha256=engine.file_hash(ROOT / "reports/overnight-v1/protocol.json"),
        data_root=load_config(ROOT / "configs/task2_real.json").data_root,
        test_audio_read=False,test_evaluated=False)
    path = REPORT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == value, "Frozen protocol changed; do not alter a launched experiment"
    else:
        write_json(path, value)
    return value


def context():
    cfg = load_config(ROOT / "configs/task2_real.json")
    metadata = read_metadata(cfg)
    assert len(metadata[metadata.group.ne("speaker:spk01")]) == 812
    state, statistics = load_source(ROOT / "runs/conformer-research")
    detector = Detector(STUDY / "assets/silero_vad.jit")
    parent = json.loads((ROOT / "reports/overnight-v1/protocol.json").read_text(encoding="utf-8"))
    sources = parent["original_training_sources"]
    recordings = {}
    for source in sources:
        path = ROOT / source["prepared_path"]
        assert engine.file_hash(path) == source["prepared_sha256"]
        recordings[source["noise_id"]] = sf.read(path, dtype="float32")[0]
    banks = {}
    for speaker in SPEAKERS:
        progress(stage="preparing_vad", speaker=speaker)
        bank = ValidationBank(speaker)
        detector.add_evaluation_bank(bank.cases, bank.raw, STUDY / "vad")
        banks[speaker] = bank
    progress(stage="preparing_pitch")
    pitch_bank, regions = prepare(metadata, cfg.data_root, STUDY / "pitch_bank")
    return dict(cfg=cfg, metadata=metadata, state=state, statistics=statistics, detector=detector,
                sources=sources, recordings=recordings, banks=banks, pitch_bank=pitch_bank,
                regions=regions, formants=LegacyFormants(ROOT / "runs/overnight-v1/waveforms", metadata))


def baseline():
    for speaker in SPEAKERS:
        out = REPORT / "folds/baseline" / f"{speaker}.json"
        if out.exists():
            continue
        source = ROOT / f"runs/overnight-v1/candidates/formant_B/{speaker}"
        parent = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        target = STUDY / "candidates/baseline" / speaker
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "history.csv", target / "history.csv")
        result = dict(candidate="baseline", spec=BASE, speaker=speaker, parameters=32122699,
                      best_epoch=parent["best_epoch"], final_epoch=100, metrics=parent["metrics"],
                      reused=True, run_dir=str(source.relative_to(ROOT)), best_sha256=engine.file_hash(source / "best.pt"))
        write_json(out, result)
    render()


def diagnostics(ctx):
    target = REPORT / "diagnostics.csv"
    if target.exists():
        return
    rows = []
    for speaker in SPEAKERS:
        bank = ctx["banks"][speaker]
        model = MaskedConformer(ctx["state"], ctx["statistics"], speaker, ctx["detector"]).cuda()
        saved = torch.load(ROOT / f"runs/overnight-v1/candidates/formant_B/{speaker}/best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        proxy = {}
        for case in bank.cases.itertuples():
            if case.kind == "noise_only" or case.true_idx == 10:
                gate = np.ones(38, np.float32)
            else:
                gate = np.full(38, .05, np.float32)
                centers = np.arange(38) * 640
                for start, end in ctx["regions"][case.path]:
                    gate[(centers >= max(0, start - 1280)) & (centers <= min(24000, end + 1280))] = 1.
            proxy[case.input_sha256] = gate
        for name, pool_only, source_proxy in (("auto_pool", True, False), ("auto_attention_pool", False, False),
                                              ("source_region_pool_diagnostic", True, True),
                                              ("source_region_attention_pool_diagnostic", False, True)):
            model.masking = True
            model.pooling_only = pool_only
            model.gate_override = (lambda wave, cache=proxy: torch.from_numpy(np.stack(
                [cache[wave_hash(w)] for w in wave.detach().cpu().numpy()])).to(wave.device)) if source_proxy else None
            scores, prediction, _ = bank.evaluate(model)
            rows.append(dict(speaker=speaker, diagnostic=name, deployable_input=not source_proxy, **scores))
            progress(stage="diagnostic", speaker=speaker, diagnostic=name, zero_f1=scores["zero_f1"])
        del model, saved
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(target, index=False)


def audit_fold(run, speaker, spec, history, optimizer):
    assert history[-1]["epoch"] == 100
    epochs = draws = 0
    for h in history[1:]:
        epoch = int(h["epoch"])
        prediction = pd.read_csv(run / "epochs" / f"epoch_{epoch:03d}.csv.gz")
        assert len(prediction) == 1827 and not prediction.path.str.contains("spk01_").any()
        for key, value in recompute_predictions(prediction).items():
            assert abs(h[key] - value) < 1e-10
        trace = pd.read_csv(run / "epochs" / f"augmentation_{epoch:03d}.csv.gz")
        assert len(trace) == 609 and trace.draw.tolist() == list(range(609))
        assert not trace.path.str.contains("spk01_|" + speaker + "_").any()
        assert not trace.speed_selected.any() and not trace.added_speech_selected.any() and not trace.near_zero_selected.any()
        if not spec["pitch"]:
            assert not trace.pitch_selected.any()
        # Match exact old source/crop/SNR draws even when the waveform augmentation changes.
        old = pd.read_csv(ROOT / f"runs/overnight-v1/candidates/formant_B/{speaker}/epochs/augmentation_{epoch:03d}.csv.gz")
        assert trace.path.tolist() == old.path.tolist()
        for column in ("noise_selected", "formant_selected", "formant_variant"):
            assert trace[column].equals(old[column]), column
        noisy = trace.noise_selected
        for column in ("noise_id", "start_sample", "snr_db"):
            assert trace.loc[noisy, column].tolist() == old.loc[noisy, column].tolist(), column
        epochs += len(prediction)
        draws += len(trace)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            assert int(optimizer.state[parameter]["step"]) == update_count(100, 16)
    return dict(epoch_predictions_recomputed=epochs, training_draws_checked=draws,
                paired_source_crop_snr_preserved=True, optimizer_steps_preserved=True)


def run_fold(spec, speaker, ctx):
    name = spec["name"]
    run = STUDY / "candidates" / name / speaker
    run.mkdir(parents=True, exist_ok=True)
    output = REPORT / "folds" / name / f"{speaker}.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    assert shutil.disk_usage(STUDY).free > 1_500_000_000, "Disk space below1.5GB; checkpoint retained"
    engine.seed_all(42)
    frame = speaker_fold(ctx["metadata"], speaker, ctx["cfg"])
    train_frame, val_frame = frame[frame.split.eq("train")], frame[frame.split.eq("val")]
    train = Training(train_frame, ctx["cfg"], speaker, ctx["sources"], [], ctx["recordings"],
                     add_speech=False, focus_zero=False, formants=ctx["formants"],
                     pitch_bank=ctx["pitch_bank"], pitch=spec["pitch"])
    train_loader, val_loader, _ = loaders(train, Clips(val_frame, ctx["cfg"]), 42, 16)
    model = MaskedConformer(ctx["state"], ctx["statistics"], speaker, ctx["detector"], masking=spec["masking"]).cuda()
    latest = run / "last.pt"
    fresh = not latest.exists()
    source = ROOT / f"runs/conformer-aug-v1/folds/{speaker}/noise/last.pt"
    checkpoint = torch.load(source if fresh else latest, map_location="cpu", weights_only=True)
    optimizer = initialize(model, checkpoint, blocks=spec["blocks"], resume=not fresh)
    bank = ctx["banks"][speaker]
    if fresh:
        assert checkpoint["epoch"] == 10
        scores, _, _ = bank.evaluate(model)
        history = [dict(epoch=10, train_loss=checkpoint["history"][-1]["train_loss"], updates=200, **scores)]
        best_epoch, best_f1 = 10, scores["zero_f1"]
        atomic_save(dict(model=model.state_dict(), epoch=10, monitor=scores), run / "best.pt")
        write_json(run / "initialization.json", dict(source=str(source.relative_to(ROOT)), source_sha256=engine.file_hash(source),
            source_epoch=10, spec=spec, parameters=sum(p.numel() for p in model.parameters()),
            kept_layers=model.kept_layers, adamw_moments_restored=True))
    else:
        history, best_f1, best_epoch = checkpoint["history"], checkpoint["best_f1"], checkpoint["best_epoch"]
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    train_loader.sampler.generator.set_state(checkpoint["loader_rng"])
    val_loader.generator.set_state(checkpoint["val_loader_rng"])
    del checkpoint
    for epoch in range(history[-1]["epoch"] + 1, 101):
        start = time.monotonic()
        train.set_epoch(epoch)
        train.trace = []
        model.train()
        total, steps = 0., 0
        for x, y, _ in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            steps += 1
            if spec["masking"]:
                gates = model.last_gates.cpu().numpy()
                for trace, gate in zip(train.trace[-len(x):], gates):
                    trace.update(vad_gate_mean=float(gate.mean()), vad_gate_max=float(gate.max()))
        assert steps == 39 and len(train.trace) == 609
        scores, prediction, _ = bank.evaluate(model)
        history.append(dict(epoch=epoch, train_loss=total / 609, updates=update_count(epoch, 16), **scores))
        folder = run / "epochs"
        folder.mkdir(exist_ok=True)
        prediction.to_csv(folder / f"epoch_{epoch:03d}.csv.gz", index=False)
        pd.DataFrame(train.trace).to_csv(folder / f"augmentation_{epoch:03d}.csv.gz", index=False)
        if scores["zero_f1"] > best_f1:
            best_f1, best_epoch = scores["zero_f1"], epoch
            atomic_save(dict(model=model.state_dict(), epoch=epoch, monitor=scores), run / "best.pt")
        atomic_save(capture(model, optimizer, history, best_f1, best_epoch, train_loader, val_loader), latest)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        progress(stage="training", candidate=name, speaker=speaker, epoch=epoch, best_epoch=best_epoch,
                 zero_f1=scores["zero_f1"], clean_f1=scores["clean_f1"], seconds=time.monotonic() - start,
                 parameters=sum(p.numel() for p in model.parameters()))
    audit = audit_fold(run, speaker, spec, history, optimizer)
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    assert checkpoint["epoch"] == best_epoch == max(history, key=lambda row: row["zero_f1"])["epoch"]
    scores, prediction, units = bank.evaluate(model, full=True)
    scores["clean_accuracy"] = float(units[units.kind.eq("clean")].accuracy.iloc[0])
    for key, value in recompute_predictions(prediction).items():
        assert abs(scores[key] - value) < 1e-10
    prediction.to_csv(run / "best_predictions.csv.gz", index=False)
    units.to_csv(run / "best_units.csv", index=False)
    _, replay, _ = bank.evaluate(model)
    joined = replay.merge(prediction, on="case_index", suffixes=("_new", "_saved"), validate="one_to_one")
    assert len(joined) == 1827 and joined.pred_idx_new.eq(joined.pred_idx_saved).all()
    assert (joined.loss_new - joined.loss_saved).abs().max() < 1e-6
    result = dict(candidate=name, spec=spec, speaker=speaker, parameters=sum(p.numel() for p in model.parameters()),
        kept_layers=model.kept_layers, best_epoch=best_epoch, final_epoch=100, metrics=scores,
        audit=dict(**audit, final_predictions_recomputed=len(prediction), best_predictions_replayed=len(replay)),
        run_dir=str(run.relative_to(ROOT)), best_sha256=engine.file_hash(run / "best.pt"), test_evaluated=False)
    write_json(run / "summary.json", result)
    write_json(output, result)
    latest.unlink()
    del model, optimizer, train, train_loader, val_loader, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    progress(stage="fold_complete", candidate=name, speaker=speaker, epoch=100)
    render()
    return result


def candidate(spec, ctx):
    path = REPORT / "candidates" / f"{spec['name']}.json"
    if path.exists():
        return
    start = time.monotonic()
    results = [run_fold(spec, speaker, ctx) for speaker in SPEAKERS]
    write_json(path, dict(spec=spec, seconds=time.monotonic() - start,
                          zero_f1=float(np.mean([row["metrics"]["zero_f1"] for row in results]))))


def results(spec):
    return [json.loads((REPORT / "folds" / spec["name"] / f"{speaker}.json").read_text(encoding="utf-8")) for speaker in SPEAKERS]


def choose():
    base = np.array([r["metrics"]["zero_f1"] for r in results(BASE)])
    all_specs = [BASE] + SPECS
    scores = {spec["name"]: np.array([r["metrics"]["zero_f1"] for r in results(spec)]) for spec in all_specs}
    eligible = [BASE] + [spec for spec in SPECS if scores[spec["name"]].mean() - base.mean() >= .005
                       and np.min(scores[spec["name"]] - base) >= -.01]
    selected = max(eligible, key=lambda spec: scores[spec["name"]].mean())
    raw = max(all_specs, key=lambda spec: scores[spec["name"]].mean())
    write_json(REPORT / "selection.json", dict(selected=selected, raw_highest=raw,
        means={key: float(value.mean()) for key, value in scores.items()}, eligible=[s["name"] for s in eligible]))
    return selected


def choose_compression(selected, compact):
    parent = results(selected)
    p = np.array([r["metrics"]["zero_f1"] for r in parent])
    original = np.mean([r["metrics"]["clean_accuracy"] for r in parent])
    eligible = [selected]
    for spec in compact:
        rows = results(spec)
        f1 = np.array([r["metrics"]["zero_f1"] for r in rows])
        clean = np.mean([r["metrics"]["clean_accuracy"] for r in rows])
        if f1.mean() >= p.mean() - .005 and np.min(f1 - p) >= -.01 and clean >= original - .005:
            eligible.append(spec)
    chosen = min(eligible, key=lambda spec: results(spec)[0]["parameters"])
    write_json(REPORT / "compression_selection.json", dict(selected=chosen,
        eligible=[spec["name"] for spec in eligible], parent=selected))
    return chosen


def listening(ctx):
    folder = REPORT / "listening"
    folder.mkdir(exist_ok=True)
    lines = ["# 발화 구간 안에서 변하는 새 피치", "", "원본 → 새 변형8종. 추가 소음·포먼트 없이 피치만 비교한다. 각1.5초 사이0.4초 간격.", ""]
    records = []
    for speaker, source in (("spk02", "audio/출발/spk02_051.wav"), ("spk03", "audio/출발/spk03_166.wav")):
        x = sf.read(Path(ctx["cfg"].data_root) / source, dtype="float32")[0]
        gap = np.zeros(6400, np.float32)
        chunks = [x]
        sf.write(folder / f"{speaker}_original.wav", x, 16000, subtype="PCM_16")
        for variant in range(8):
            wave = ctx["pitch_bank"].get(source, variant, 0)
            name = f"{speaker}_pitch_{variant}.wav"
            sf.write(folder / name, wave, 16000, subtype="PCM_16")
            chunks.extend((gap, wave))
            records.append(dict(source=source, variant=variant, file=name, input_sha256=wave_hash(wave)))
        sf.write(folder / f"{speaker}_comparison.wav", np.concatenate(chunks), 16000, subtype="PCM_16")
        lines.append(f"- [{speaker} 원본→8종]({speaker}_comparison.wav)")
    (folder / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(folder / "manifest.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    STUDY.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    with (STUDY / "driver.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        engine.seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        assert torch.cuda.is_available()
        frozen = freeze()
        if (STUDY / "complete.json").exists():
            return
        ctx = context()
        if args.preflight:
            from .checks import preflight
            preflight(ctx, ROOT, STUDY, REPORT)
            checks = json.loads((REPORT / "preflight.json").read_text(encoding="utf-8"))
            checks["source_sha256"] = frozen["source_sha256"]
            write_json(REPORT / "preflight.json", checks)
            listening(ctx)
            render()
            progress(stage="ready_to_launch")
            return
        assert json.loads((REPORT / "preflight.json").read_text(encoding="utf-8"))["source_sha256"] == frozen["source_sha256"]
        (STUDY / "failure.json").unlink(missing_ok=True)
        process = STUDY / "process.json"
        started = json.loads(process.read_text(encoding="utf-8"))["started_unix"] if process.exists() else time.time()
        write_json(process, dict(pid=os.getpid(), started_unix=started, deadline_unix=started + 7 * 3600,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), session_id=os.getsid(0)))
        baseline()
        diagnostics(ctx)
        for spec in SPECS:
            candidate(spec, ctx)
        selected = choose()
        compact = []
        skips = []
        for blocks in (8, 6):
            spec = dict(selected, name=f"compact{blocks}", blocks=blocks)
            done = (REPORT / "candidates" / f"compact{blocks}.json").exists()
            durations = [json.loads(p.read_text(encoding="utf-8"))["seconds"] for p in (REPORT / "candidates").glob("*.json")]
            estimate = max(max(durations, default=0), 45 * 60)
            remaining = started + 7 * 3600 - time.time()
            if not done and remaining < estimate + 20 * 60:
                skips.append(dict(blocks=blocks, remaining_seconds=remaining,
                                  reason="Optional only; reserve completed4fold and final audit within7hour target"))
                write_json(REPORT / "optional_skips.json", skips)
                continue
            candidate(spec, ctx)
            compact.append(spec)
        chosen = choose_compression(selected, compact)
        progress(stage="auditing", candidate=chosen["name"])
        records = [json.loads(p.read_text(encoding="utf-8")) for p in (REPORT / "folds").glob("*/*.json")]
        for row in records:
            assert engine.file_hash(ROOT / row["run_dir"] / "best.pt") == row["best_sha256"]
            assert row["final_epoch"] == 100
        assert freeze() == frozen
        audits = [r["audit"] for r in records if not r.get("reused")]
        write_json(REPORT / "independent_audit.json", dict(trained_folds=len(audits),
            epoch_predictions_recomputed=sum(r["epoch_predictions_recomputed"] for r in audits),
            training_draws_checked=sum(r["training_draws_checked"] for r in audits),
            final_predictions_recomputed=sum(r["final_predictions_recomputed"] for r in audits),
            source_hashes_preserved=True, paired_source_crop_snr_preserved=True,
            all_trained_folds_to100=True, test_audio_read=False, test_evaluated=False))
        write_json(STUDY / "complete.json", dict(selected_full=selected, selected_deployment=chosen,
            optional_skips=skips, all_stages_and_audits_complete=True, elapsed_seconds=time.time() - started,
            test_evaluated=False))
        render()
        progress(stage="complete", candidate=chosen["name"])
        print("ATTENTION_PITCH_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if STUDY.exists():
            write_json(STUDY / "failure.json", dict(error=traceback.format_exc(), timestamp=time.time(), test_evaluated=False))
        raise
