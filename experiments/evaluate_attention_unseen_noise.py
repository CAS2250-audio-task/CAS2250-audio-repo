"""Frozen attention checkpoints on new noise recordings, without filters or training."""
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from src import engine
from src.config import load_config
from experiments.conformer.model import load_source
from experiments.attention_pitch.model import MaskedConformer
from experiments.attention_pitch.vad import Detector, wave_hash
from experiments.noise_stress.mixing import crop_noise, mix_at_snr


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/attention-unseen-noise-v1"
REPORT = ROOT / "reports/attention-unseen-noise-v1"
SPEAKERS = ("spk02", "spk03", "spk04", "spk05")
FAMILIES = dict(typing="키보드 타이핑", door="문 닫힘",
                neighbor_speech="옆 사람 말소리", vacuum="청소기")
UNIT_KEYS = ["suite", "heldout_speaker", "kind", "noise_id", "family", "snr_db", "draw"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    engine.write_json(temp, value)
    temp.replace(path)


def progress(**values):
    values.update(pid=os.getpid(), updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json(STUDY / "progress.json", values)
    print(json.dumps(values, ensure_ascii=False), flush=True)


def prepare_sources():
    sources = json.loads((STUDY / "source_assets.json").read_text(encoding="utf-8"))
    assert len(sources) == 7 and set(s["family"] for s in sources) == set(FAMILIES)
    old = []
    for name in ("conformer-aug-v1/training_noise_sources.json", "noise-stress-v1/noise_sources.json"):
        old.extend(json.loads((ROOT / "reports" / name).read_text(encoding="utf-8")))
    seen = set()
    for source in sources:
        original = ROOT / source["original_path"]
        assert engine.file_hash(original) == source["original_sha256"]
        assert source["original_sha256"] not in {s["original_sha256"] for s in old}
        assert original.name not in {s["original_filename"] for s in old}
        audio, rate = sf.read(original, dtype="float32", always_2d=True)
        wave = torch.from_numpy(audio.mean(1))
        if rate != 16000:
            wave = torchaudio.functional.resample(wave, rate, 16000)
        assert wave.ndim == 1 and len(wave) >= 24000 and torch.isfinite(wave).all()
        target = STUDY / "sources" / (source["noise_id"] + ".wav")
        if target.exists():
            saved, sr = sf.read(target, dtype="float32")
            assert sr == 16000 and np.array_equal(saved, wave.numpy())
        else:
            sf.write(target, wave.numpy(), 16000, subtype="FLOAT")
        digest = wave_hash(wave.numpy())
        assert digest not in seen
        seen.add(digest)
        source.update(original_rate=rate, original_channels=audio.shape[1],
                      prepared_path=str(target.relative_to(ROOT)),
                      prepared_sha256=engine.file_hash(target), pcm_sha256=digest,
                      seconds=len(wave) / 16000)
        assert source["prepared_sha256"] not in {s["prepared_sha256"] for s in old}
    write_json(REPORT / "noise_sources.json", sources)
    return sources


def freeze_protocol(sources):
    parent_path = ROOT / "reports/attention-pitch-v1/protocol.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    pins = dict(parent["source_sha256"])
    pins[str(Path(__file__).relative_to(ROOT))] = engine.file_hash(Path(__file__))
    checkpoints = {}
    banks = {}
    for speaker in SPEAKERS:
        result = json.loads((ROOT / f"reports/attention-pitch-v1/folds/attention/{speaker}.json").read_text())
        checkpoint = ROOT / result["run_dir"] / "best.pt"
        checkpoints[str(checkpoint.relative_to(ROOT))] = result["best_sha256"]
        for filename in (f"{speaker}_inputs.npy", f"{speaker}_cases.csv.gz"):
            path = ROOT / "runs/denoise-eval-v1" / filename
            banks[str(path.relative_to(ROOT))] = engine.file_hash(path)
        path = ROOT / f"runs/attention-pitch-v1/vad/{speaker}.npy"
        banks[str(path.relative_to(ROOT))] = engine.file_hash(path)
    vad_path = ROOT / "runs/attention-pitch-v1/assets/silero_vad.jit"
    pins[str(vad_path.relative_to(ROOT))] = parent["vad"]["sha256"]
    for path, expected in {**pins, **checkpoints}.items():
        assert engine.file_hash(ROOT / path) == expected, path
    result = dict(study="attention-unseen-noise-v1", model="attention", filters="none",
                  training=False, checkpoint_selection="unchanged prior attention best per development speaker",
                  speakers=list(SPEAKERS), sample_rate=16000, samples=24000,
                  snr_db=[0], draws=3, noise_only_windows=20, classifier_batch=32,
                  batch_boundaries="reset each 203-clip mixed or clean unit and 20-clip noise-only unit",
                  vad="unchanged Silero on the actual noisy waveform; no source-side boundaries",
                  signal_reference="full original 1.5-second waveform power, including original background",
                  silence_reference="median keyword RMS of the other three development speakers, squared",
                  mixing="unchanged deterministic crop_noise/mix_at_snr; shared attenuation to peak .99",
                  new_noise_sources=sources, existing_noise_sources=8,
                  reporting="macro-F1 over all 11 labels per speaker/source/draw; report source and family means separately",
                  no_tuning="sources fixed before inference; no filter, threshold, checkpoint, or hyperparameter search",
                  source_sha256=pins, checkpoint_sha256=checkpoints, bank_sha256=banks,
                  parent_protocol_sha256=engine.file_hash(parent_path),
                  test_audio_read=False, test_evaluated=False,
                  limitations=["new noise recordings on previously used development speakers, not a new-speaker test",
                               "seven files are not claimed to be seven independent recording sessions",
                               "three crops are not independent training seeds",
                               "file/hash separation does not prove absence of partial underlying-recording overlap",
                               "no claim that these noises were unseen in original ASR pretraining"])
    path = REPORT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == result, "Frozen protocol changed"
    else:
        write_json(path, result)
    return result


def clean_banks():
    originals = {}
    for speaker in SPEAKERS:
        cases = pd.read_csv(ROOT / f"runs/denoise-eval-v1/{speaker}_cases.csv.gz")
        assert len(cases) == 5235 and cases.heldout_speaker.eq(speaker).all()
        assert not cases.path.str.contains("spk01_").any()
        raw = np.load(ROOT / f"runs/denoise-eval-v1/{speaker}_inputs.npy", mmap_mode="r")
        assert raw.shape == (5235, 24000)
        clean = cases[cases.kind.eq("clean")].reset_index(drop=True)
        waves = np.array(raw[clean.case_index.to_numpy()])
        assert len(clean) == 203
        assert [wave_hash(w) for w in waves] == clean.input_sha256.tolist()
        originals[speaker] = (clean, waves)
    return originals


@torch.inference_mode()
def evaluate_unit(model, frame, waves):
    assert waves.shape == (len(frame), 24000) and waves.dtype == np.float32
    assert np.isfinite(waves).all() and not frame.path.str.contains("spk01_").any()
    assert [wave_hash(w) for w in waves] == frame.input_sha256.tolist()
    predictions, losses = [], []
    for offset in range(0, len(frame), 32):
        chunk = frame.iloc[offset:offset + 32]
        x = torch.from_numpy(waves[offset:offset + 32].copy()).cuda()
        logits = model(x)
        assert torch.isfinite(logits).all()
        targets = torch.tensor(chunk.true_idx.to_numpy(), device="cuda")
        predictions.extend(logits.argmax(1).cpu().tolist())
        losses.extend(F.cross_entropy(logits, targets, reduction="none").cpu().tolist())
    output = frame.copy()
    output["pred_idx"], output["loss"] = predictions, losses
    return output


def new_unit(source, recording, clean, original, speaker, reference, draw, only=False):
    base = pd.DataFrame(dict(path=[f"noise-only/{i:02d}" for i in range(20)], true_idx=10)) if only else clean
    rows, waves = [], []
    for i, row in enumerate(base.itertuples()):
        noise, crop = crop_noise(recording, "noise-only" if only else row.path,
                                 source["noise_id"], i if only else draw)
        wave = np.zeros(24000, np.float32) if only else original[i]
        power = reference if int(row.true_idx) == 10 else float(np.mean(wave.astype(np.float64) ** 2))
        mixed, audit = mix_at_snr(wave, noise, 0, power)
        record = {**crop, **audit}
        record.update(suite="new", heldout_speaker=speaker,
                      kind="noise_only" if only else "mixed", noise_id=source["noise_id"],
                      family=source["family"], draw=0 if only else draw, crop_draw=crop["draw"],
                      path=row.path, true_idx=int(row.true_idx),
                      input_sha256=record.pop("mixture_sha256"))
        rows.append(record)
        waves.append(mixed)
    return pd.DataFrame(rows), np.stack(waves)


def summarize(predictions):
    rows = []
    for keys, group in predictions.groupby(UNIT_KEYS, sort=False):
        rows.append(dict(zip(UNIT_KEYS, keys), **engine.metrics(group.true_idx, group.pred_idx, 11),
                         loss=float(group.loss.mean()), cases=len(group)))
    units = pd.DataFrame(rows)
    units.to_csv(REPORT / "units.csv", index=False)
    mixed = units[units.kind.eq("mixed")]
    noise = mixed.groupby(["suite", "family", "noise_id"], as_index=False).agg(
        f1=("macro_f1", "mean"), accuracy=("accuracy", "mean"), worst_unit_f1=("macro_f1", "min"))
    only = predictions[predictions.kind.eq("noise_only")].copy()
    only["false_positive"] = only.pred_idx.ne(10)
    noise = noise.merge(only.groupby(["suite", "family", "noise_id"], as_index=False).false_positive.mean(),
                        on=["suite", "family", "noise_id"])
    noise.to_csv(REPORT / "noise_summary.csv", index=False)
    speaker = mixed.groupby(["suite", "family", "heldout_speaker"], as_index=False).agg(
        f1=("macro_f1", "mean"), accuracy=("accuracy", "mean"))
    speaker.to_csv(REPORT / "speaker_family.csv", index=False)
    new = noise[noise.suite.eq("new")]
    family = new.groupby("family", as_index=False).agg(
        f1=("f1", "mean"), accuracy=("accuracy", "mean"),
        noise_only_fpr=("false_positive", "mean"), recordings=("noise_id", "size"))
    spk02 = speaker[speaker.suite.eq("new") & speaker.heldout_speaker.eq("spk02")][["family", "f1"]]
    family = family.merge(spk02.rename(columns={"f1": "spk02_f1"}), on="family")
    family.to_csv(REPORT / "family_summary.csv", index=False)
    clean = units[units.kind.eq("clean")]
    summary = dict(existing_f1=float(noise[noise.suite.eq("existing")].f1.mean()),
                   new_recording_mean_f1=float(new.f1.mean()),
                   new_family_mean_f1=float(family.f1.mean()),
                   clean_f1=float(clean.macro_f1.mean()),
                   new_noise_only_fpr=float(only[only.suite.eq("new")].false_positive.mean()),
                   prediction_rows=len(predictions), test_evaluated=False)
    write_json(REPORT / "summary.json", summary)
    lines = ["# Attention 최고 모델: 새로운 소음 일반화 확인", "",
             "관측 최고 attention의 기존 4개 개발 화자 checkpoint를 고정하고, 필터·재학습 없이 평가했다.",
             "16kHz·1.5초, 기존 정의의 0dB, 소음별 3개 추출. test01은 읽거나 평가하지 않았다.", "",
             f"기존 8종 F1 재현: **{summary['existing_f1']*100:.4f}%**, 원본 F1 {summary['clean_f1']*100:.4f}%.",
             f"새 4유형 동일가중 평균 F1: **{summary['new_family_mean_f1']*100:.4f}%**; "
             f"7개 파일 동일가중 평균 {summary['new_recording_mean_f1']*100:.4f}%.", "",
             "| 새 소음 유형 | 파일 수 | 0dB F1 | 정확도 | 화자02 F1 | 잡음 단독 오반응 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in family.sort_values("f1", ascending=False).itertuples():
        lines.append(f"| {FAMILIES[r.family]} | {r.recordings} | {r.f1*100:.2f}% | {r.accuracy*100:.2f}% | {r.spk02_f1*100:.2f}% | {r.noise_only_fpr*100:.2f}% |")
    lines += ["", "기존과 새 소음 집합의 난이도가 달라 두 평균의 차이를 모델 개선·퇴보량으로 해석하지 않는다.",
              "새 소음 평가는 기존 개발 화자에 대한 평가이며, 새로운 화자에 대한 독립 test가 아니다.",
              "타이핑·문 닫힘·말소리는 각 2파일, 청소기는 1파일이다. 번호가 다른 파일이 독립 녹음 세션임을 보장하지 않는다.",
              "이전 분류기 훈련 6개 및 기존 평가 8개의 원본·준비 파일 해시와 겹치지 않음을 확인했다. 원본 ASR 사전학습까지 완전히 미관측이라는 주장은 하지 않는다.",
              "0dB는 원본 전체 파형 파워 대 추가 소음 파워다. silence와 잡음 단독은 fold 훈련 명령어 RMS 중앙값 기준이다.",
              "소음 단독은 파일별·화자별 20개 창이며, 발화가 있는 혼합 F1과 별도로 본다.", "",
              "[MS-SNSD 공식 소음 출처](https://github.com/microsoft/MS-SNSD/tree/fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742/noise_test)",
              "[고정 프로토콜](protocol.json) · [원본 출처](noise_sources.json) · [파일별](noise_summary.csv) · [화자별](speaker_family.csv) · [검증](audit.json)", "",
              "![소음 유형별 화자 F1](speaker_noise.png)", ""]
    (REPORT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = speaker[speaker.suite.eq("new")].pivot(index="family", columns="heldout_speaker", values="f1")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(table.to_numpy()*100, vmin=40, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(4), table.columns)
    ax.set_yticks(range(len(table)), table.index)
    for y in range(len(table)):
        for x in range(4):
            value = table.iloc[y, x]*100
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value > 80 else "black")
    ax.set_title("Frozen attention model: new noises at 0 dB (macro-F1 %)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(REPORT / "speaker_noise.png", dpi=160)
    plt.close(fig)
    return summary, units


@torch.inference_mode()
def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    STUDY.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = (STUDY / "driver.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (STUDY / "complete.json").exists():
        print("ALREADY_COMPLETE", flush=True)
        return
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    engine.seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sources = prepare_sources()
    protocol = freeze_protocol(sources)
    originals = clean_banks()
    recordings = {s["noise_id"]: sf.read(ROOT / s["prepared_path"], dtype="float32")[0] for s in sources}
    state, statistics = load_source(ROOT / "runs/conformer-research")
    detector = Detector(ROOT / "runs/attention-pitch-v1/assets/silero_vad.jit")
    parts = STUDY / "parts"
    parts.mkdir(exist_ok=True)
    replayed = 0
    for speaker in SPEAKERS:
        progress(stage="speaker_start", speaker=speaker, elapsed_seconds=time.monotonic()-started)
        checkpoint = ROOT / f"runs/attention-pitch-v1/candidates/attention/{speaker}/best.pt"
        model = MaskedConformer(state, statistics, speaker, detector, masking=True).cuda().eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
        cases = pd.read_csv(ROOT / f"runs/denoise-eval-v1/{speaker}_cases.csv.gz")
        raw = np.load(ROOT / f"runs/denoise-eval-v1/{speaker}_inputs.npy", mmap_mode="r")
        gates = np.load(ROOT / f"runs/attention-pitch-v1/vad/{speaker}.npy")
        assert gates.shape == (5235, 38)
        detector.evaluation_cache = dict(zip(cases.input_sha256, gates))
        prior = pd.read_csv(checkpoint.parent / "best_predictions.csv.gz").set_index("case_index")
        replay_path = parts / f"existing_{speaker}.csv.gz"
        if not replay_path.exists():
            outputs = []
            for _, group in cases.groupby(["kind", "noise_id", "snr_db", "draw"], sort=False):
                frame = group.copy()
                frame["suite"], frame["family"] = "existing", frame.noise_id
                output = evaluate_unit(model, frame, np.array(raw[frame.case_index.to_numpy()]))
                expected = prior.loc[output.case_index]
                assert output.pred_idx.tolist() == expected.pred_idx.tolist(), "Prior attention predictions changed"
                outputs.append(output)
            temporary = replay_path.with_suffix(".partial.gz")
            pd.concat(outputs, ignore_index=True).to_csv(temporary, index=False)
            temporary.replace(replay_path)
        replayed += len(pd.read_csv(replay_path))
        clean, original = originals[speaker]
        rms = []
        for other, (frame, waves) in originals.items():
            if other != speaker:
                rms.extend(np.sqrt(np.mean(waves[frame.true_idx.ne(10)].astype(np.float64)**2, axis=1)).tolist())
        assert len(rms) == 600
        reference = float(np.median(rms)**2)
        for source in sources:
            target = parts / f"{speaker}_{source['noise_id']}.csv.gz"
            if not target.exists():
                outputs = []
                for draw in range(3):
                    frame, waves = new_unit(source, recordings[source["noise_id"]], clean, original,
                                            speaker, reference, draw)
                    outputs.append(evaluate_unit(model, frame, waves))
                frame, waves = new_unit(source, recordings[source["noise_id"]], clean, original,
                                        speaker, reference, 0, only=True)
                outputs.append(evaluate_unit(model, frame, waves))
                temporary = target.with_suffix(".partial.gz")
                pd.concat(outputs, ignore_index=True).to_csv(temporary, index=False)
                temporary.replace(target)
            progress(stage="noise_done", speaker=speaker, noise=source["noise_id"],
                     completed_source_folds=len(list(parts.glob("spk*.csv.gz"))),
                     elapsed_seconds=time.monotonic()-started)
        del model, raw
        gc.collect()
        torch.cuda.empty_cache()
    predictions = pd.concat([pd.read_csv(p) for p in sorted(parts.glob("*.csv.gz"))], ignore_index=True)
    assert len(predictions) == 38552 and replayed == 20940
    assert not predictions.path.str.contains("spk01_").any()
    predictions.to_csv(REPORT / "predictions.csv.gz", index=False)
    summary, units = summarize(predictions)
    prior_summary = pd.read_csv(ROOT / "reports/attention-pitch-v1/summary.csv")
    assert abs(summary["existing_f1"]-float(prior_summary.loc[prior_summary.candidate.eq("attention"), "zero_f1"].iloc[0])) < 1e-12
    for group in ("source_sha256", "checkpoint_sha256", "bank_sha256"):
        for path, digest in protocol[group].items():
            assert engine.file_hash(ROOT / path) == digest, path
    for source in sources:
        for field in ("original", "prepared"):
            assert engine.file_hash(ROOT / source[f"{field}_path"]) == source[f"{field}_sha256"]
    audit = dict(existing_attention_predictions_exactly_replayed=replayed,
                 new_mixed_predictions=17052, new_noise_only_predictions=560,
                 prediction_rows=len(predictions), units=len(units),
                 original_checkpoint_and_sources_preserved=True,
                 new_mixture_hashes_checked=17612,
                 filters_used=False, training_performed=False, test_audio_read=False, test_evaluated=False,
                 elapsed_seconds=time.monotonic()-started)
    write_json(REPORT / "audit.json", audit)
    write_json(STUDY / "complete.json", {**summary, **audit})
    progress(stage="complete", **summary, elapsed_seconds=time.monotonic()-started)


if __name__ == "__main__":
    main()
