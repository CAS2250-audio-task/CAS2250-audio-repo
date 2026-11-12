"""Preserve raw context in 1.5-second audio, with explicit short-source handling.

The default produces continuous candidates and leaves short originals pending.
With --short-policy zero, retain each short original in full and pad only its
missing length. The reviewed dataset and its recipe stay intact in both modes.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000
SAMPLES = 24000
IDENTITY = ("path", "label", "group", "split", "speaker_id", "room_session_id")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fields=None):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audio_path(root, relative):
    rel = Path(relative)
    result = (root / rel).resolve()
    if rel.is_absolute() or not result.is_relative_to(root) or rel.suffix.lower() != ".wav":
        raise ValueError(f"Expected a relative WAV inside the dataset: {relative}")
    if not result.is_file():
        raise ValueError(f"Missing audio: {result}")
    return result


def continuous_window(source_frames, old_start, old_end, old_left_padding):
    """Keep the entire reviewed crop, including its original safety margins."""
    if not 0 <= old_start < old_end <= source_frames or old_end - old_start > SAMPLES:
        raise ValueError("Invalid reviewed crop")
    if old_left_padding < 0 or old_left_padding > SAMPLES - (old_end - old_start):
        raise ValueError("Invalid reviewed padding")
    if source_frames < SAMPLES:
        return None
    start = min(max(0, old_start - old_left_padding), source_frames - SAMPLES)
    end = start + SAMPLES
    if start > old_start or end < old_end:
        raise ValueError("A continuous window would discard reviewed audio")
    return start, end


def prepare_candidates(raw_root, reviewed_root, output_root, *, short_policy="pending"):
    if short_policy not in ("pending", "zero"):
        raise ValueError("short_policy must be pending or zero")
    raw, reviewed, output = map(lambda p: Path(p).resolve(),
                                (raw_root, reviewed_root, output_root))
    for source in (raw, reviewed):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output must be separate from both source datasets")
    if output.exists():
        raise FileExistsError(f"Use a fresh output folder: {output}")
    metadata_file = reviewed / "metadata.csv"
    recipe_file = reviewed / "preprocessing_recipe.csv"
    metadata, recipe, raw_metadata = map(read_rows,
        (metadata_file, recipe_file, raw / "metadata.csv"))
    indexed = {}
    for title, rows in (("metadata", metadata), ("recipe", recipe), ("raw", raw_metadata)):
        indexed[title] = {r["path"]: r for r in rows}
        if len(indexed[title]) != len(rows):
            raise ValueError(f"Duplicate paths in {title}")
    if not metadata or not (indexed["metadata"].keys() == indexed["recipe"].keys()
                            == indexed["raw"].keys()):
        raise ValueError("Raw, reviewed and recipe file identities must match")

    tracked = {p: digest(p) for p in (metadata_file, recipe_file, raw / "metadata.csv")}
    plans = []
    for r in recipe:
        for field in IDENTITY:
            if any(r[field] != indexed[key][r["path"]][field] for key in ("metadata", "raw")):
                raise ValueError(f"Changed identity {field}: {r['path']}")
        source = audio_path(raw, r["path"])
        previous = audio_path(reviewed, r["path"])
        for file, expected in ((source, r["raw_wav_sha256"]), (previous, r["output_sha256"])):
            actual = digest(file)
            if actual != expected:
                raise ValueError(f"Source hash mismatch: {file}")
            tracked[file] = actual
        info = sf.info(source)
        previous_info = sf.info(previous)
        if (info.samplerate, info.channels, info.subtype) != (SAMPLE_RATE, 1, "PCM_16"):
            raise ValueError(f"Unexpected raw WAV format: {source}")
        if (previous_info.samplerate, previous_info.channels, previous_info.subtype,
                previous_info.frames) != (SAMPLE_RATE, 1, "PCM_16", SAMPLES):
            raise ValueError(f"Unexpected reviewed WAV format: {previous}")
        n = int(r["source_frames"])
        a, b = int(r["crop_start_sample"]), int(r["crop_end_sample"])
        left, right = int(r["pad_left_samples"]), int(r["pad_right_samples"])
        gain = float(r["gain_db"])
        if n != info.frames or min(left, right) < 0 or b - a + left + right != SAMPLES:
            raise ValueError(f"Invalid recipe length: {r['path']}")
        if not np.isfinite(gain) or (r["label"] == "silence" and gain != 0):
            raise ValueError(f"Invalid recipe gain: {r['path']}")
        window = continuous_window(n, a, b, left)
        plan = {field: r[field] for field in IDENTITY}
        status = ("continuous_candidate" if window else
                  "zero_padded_original" if short_policy == "zero" else "pending_short_original")
        plan.update(source_frames=n, source_seconds=n / SAMPLE_RATE,
                    previous_crop_start_sample=a, previous_crop_end_sample=b,
                    previous_pad_left_samples=left, previous_pad_right_samples=right,
                    previous_gain_db=gain, raw_wav_sha256=r["raw_wav_sha256"],
                    previous_output_sha256=r["output_sha256"],
                    previous_method=r.get("method", ""),
                    status=status,
                    crop_start_sample="", crop_end_sample="", gain_db="",
                    pad_left_samples=0, pad_right_samples=0,
                    added_left_context_samples=0, added_right_context_samples=0,
                    peak_gain_reduced=False, review_required=False, output_sha256="")
        if window or short_policy == "zero":
            start, end = window if window else (0, n)
            remaining = SAMPLES - (end - start)
            pad_left = remaining // 2
            signal, _ = sf.read(source, start=start, stop=end, dtype="float32")
            if not np.isfinite(signal).all() or signal.shape != (end - start,):
                raise ValueError(f"Invalid raw waveform: {source}")
            peak = float(np.abs(signal).max())
            # Extending context can expose a louder transient. Lower the whole
            # clip's existing gain only when needed; never hard-clip samples.
            applied = gain if r["label"] == "silence" else min(
                gain, -1.0 - 20 * np.log10(max(peak, 1e-9)))
            reduced = applied < gain - 1e-6
            plan.update(crop_start_sample=start, crop_end_sample=end, gain_db=applied,
                        pad_left_samples=pad_left, pad_right_samples=remaining - pad_left,
                        added_left_context_samples=a - start,
                        added_right_context_samples=end - b,
                        peak_gain_reduced=reduced, review_required=reduced)
        plans.append(plan)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".real-context-", dir=output.parent) as folder:
        stage = Path(folder) / "candidates"
        stage.mkdir()
        for plan in plans:
            if plan["status"] == "pending_short_original":
                continue
            destination = stage / plan["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (plan["previous_pad_left_samples"] == plan["previous_pad_right_samples"] == 0
                    and not plan["peak_gain_reduced"]):
                shutil.copyfile(reviewed / plan["path"], destination)
            else:
                signal, _ = sf.read(raw / plan["path"], dtype="float32",
                                    start=plan["crop_start_sample"], stop=plan["crop_end_sample"])
                signal = signal * (10 ** (plan["gain_db"] / 20))
                if float(np.abs(signal).max()) >= 1:
                    raise ValueError(f"Output would clip: {plan['path']}")
                signal = np.pad(signal, (plan["pad_left_samples"], plan["pad_right_samples"]))
                sf.write(destination, signal, SAMPLE_RATE, subtype="PCM_16")
            plan["output_sha256"] = digest(destination)
        changed = [str(file) for file, before in tracked.items() if digest(file) != before]
        if changed:
            raise ValueError(f"Input files changed during preparation: {changed[:3]}")
        accepted = {p["path"] for p in plans if p["status"] != "pending_short_original"}
        pending = [p for p in plans if p["status"] == "pending_short_original"]
        complete = short_policy == "zero"
        metadata_name = "metadata.csv" if complete else "candidate_metadata.csv"
        recipe_name = "preprocessing_recipe.csv" if complete else "context_recipe.csv"
        if complete:
            shutil.copyfile(metadata_file, stage / metadata_name)
        else:
            write_rows(stage / metadata_name, [r for r in metadata if r["path"] in accepted],
                       list(metadata[0]))
        write_rows(stage / recipe_name, plans)
        write_rows(stage / "pending_short_sources.csv", pending, list(plans[0]))
        manifest = dict(dataset_version="task2-real-context-1p5-v1" if complete
                        else "real-context-1p5-candidates-v1",
            status="prepared" if complete else "pending_short_source_policy" if pending else "candidates_ready",
            sample_rate=SAMPLE_RATE, samples=SAMPLES, seconds=1.5,
            format="mono PCM16 WAV", files=len(accepted), total_originals=len(plans),
            short_source_policy=short_policy,
            continuous_candidates=sum(p["status"] == "continuous_candidate" for p in plans),
            zero_padded_originals=sum(p["status"] == "zero_padded_original" for p in plans),
            pending_short_originals=len(pending),
            pending_by_split=dict(Counter(p["split"] for p in pending)),
            pending_by_group=dict(Counter(p["group"] for p in pending)),
            candidate_split_counts=dict(Counter(p["split"] for p in plans if p["path"] in accepted)),
            split_counts=dict(Counter(p["split"] for p in plans if p["path"] in accepted)),
            gain_reduced_files=[p["path"] for p in plans if p["peak_gain_reduced"]],
            artificial_padding_added=sum(p["pad_left_samples"] + p["pad_right_samples"] for p in plans),
            time_stretching_applied=False, listening_review_status="not_performed",
            all_reviewed_crop_samples_retained=True, inputs_unchanged=True,
            input_wavs_verified=sum(p.suffix == ".wav" for p in tracked),
            raw_metadata_sha256=digest(raw / "metadata.csv"),
            metadata_sha256=digest(stage / metadata_name), recipe_sha256=digest(stage / recipe_name),
            reviewed_metadata_sha256=digest(metadata_file), reviewed_recipe_sha256=digest(recipe_file),
            raw_root=str(raw), reviewed_root=str(reviewed))
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if complete:
            (stage / "README.md").write_text(
                "# 원본 주변 녹음을 보존한 1.5초 데이터\n\n"
                f"전체 {len(plans)}개, mono PCM16 / 16 kHz / 24,000샘플입니다.\n\n"
                f"원본이 충분한 {manifest['continuous_candidates']}개는 검수 구간 전체를 포함한 "
                "원본 연속 1.5초를 사용했습니다.\n"
                f"짧은 {manifest['zero_padded_originals']}개는 원본 전체를 사용하고 부족한 길이만 "
                "좌우 0값 패딩으로 채웠습니다. 홀수 패딩은 오른쪽에 한 샘플 더 넣습니다.\n\n"
                "원본·기존 검수본·라벨·화자·split은 보존했습니다. 배경음 반복, 속도 변환, "
                "잡음 증강은 적용하지 않았습니다. 기존 gain을 유지하되 새 주변음 때문에 "
                "피크 상한 -1 dBFS를 넘으면 전체 gain을 낮추고 review_required에 기록했습니다. "
                "silence는 gain 0 dB를 유지했습니다.\n\n"
                "자동 파일 검증은 완료했으며 새 주변음의 청취 검수는 아직 하지 않았습니다. "
                "preprocessing_recipe.csv에 원본 해시, 구간, 패딩, gain과 출력 해시가 있습니다.\n")
        else:
            (stage / "README.md").write_text(
                "# 원본 연속 1.5초 후보본\n\n"
                f"전체 {len(plans)}개 중 {len(accepted)}개를 원본의 연속 24,000샘플로 만들었습니다.\n"
                f"원본 자체가 짧은 {len(pending)}개는 처리 방식 선택을 기다립니다.\n\n"
                "현재 폴더는 완성된 학습 데이터셋이 아닙니다. candidate_metadata.csv는 후보 파일 목록이며, "
                "짧은 파일을 제외한 채 학습·평가하면 화자와 라벨 분포가 달라집니다.\n\n"
                "기존 검수 구간을 통째로 포함하며 주변 녹음을 확장했습니다. 0값 패딩·반복·속도 변환은 "
                "추가하지 않았습니다. 원본 녹음 자체에 있는 0값 샘플은 그대로 유지합니다.\n"
                "기존 gain을 유지하되 확장 구간 때문에 피크 상한 -1 dBFS를 넘으면 전체 gain을 낮추고 "
                "review_required에 표시합니다. silence는 gain 0 dB를 유지합니다.\n")
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--reviewed-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--short-policy", choices=("pending", "zero"), default="pending")
    args = parser.parse_args()
    print(json.dumps(prepare_candidates(args.raw_root, args.reviewed_root, args.output_root,
                                        short_policy=args.short_policy),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
