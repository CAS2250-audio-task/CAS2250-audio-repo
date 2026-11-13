"""Export exploratory pitch/formant listening candidates without changing training."""
import hashlib
import html
import json
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf

from experiments.overnight.data import shift_formants
from experiments.overnight.waveforms import pitch_wave


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/overnight-v1/listening/pitch-formant-diversity"
RATE = 16000
SAMPLES = 24000
FORMANTS = (.8, .9, .95, 1.05, 1.1, 1.25)


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_wave(wave):
    return hashlib.sha256(np.asarray(wave, dtype=np.float32).tobytes()).hexdigest()


def level_match(wave, original):
    wave = np.asarray(wave, dtype=np.float64)
    rms = float(np.sqrt(np.mean(wave * wave)))
    reference = float(np.sqrt(np.mean(original.astype(np.float64) ** 2)))
    gain = reference / rms if rms else 1.
    peak = float(np.max(np.abs(wave)))
    gain = min(gain, .99 / peak) if peak else gain
    return (wave * gain).astype(np.float32), gain


def save_wave(path, wave):
    wave = np.asarray(wave, dtype=np.float32)
    assert wave.ndim == 1 and np.isfinite(wave).all()
    assert np.max(np.abs(wave)) <= 1.
    sf.write(path, wave, RATE, subtype="PCM_16")
    restored, rate = sf.read(path, dtype="float32")
    assert rate == RATE and restored.shape == wave.shape
    assert np.max(np.abs(restored - wave)) <= 1 / 32768 + 1e-7
    return dict(file=str(path.relative_to(REPORT)), samples=len(wave),
                duration_seconds=len(wave) / RATE, file_sha256=sha_file(path),
                float32_sha256=sha_wave(wave), peak=float(np.max(np.abs(wave))),
                rms=float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))))


def recipes():
    items = []
    for ratio in FORMANTS:
        tag = str(ratio).replace(".", "p")
        items.append(dict(name=f"formant_{tag}", group="formant",
                          label=f"포먼트 {ratio:.2f}배 / 피치 유지", ratio=ratio, curve=None))
    for amplitude in (1, 2, 3):
        for direction in (-1, 1):
            curve = [0., float(direction * amplitude), float(-direction * amplitude), 0.]
            tag = "down_up" if direction == -1 else "up_down"
            items.append(dict(name=f"pitch_{amplitude}st_{tag}", group="pitch", ratio=1.,
                              label=f"피치 0→{curve[1]:+g}→{curve[2]:+g}→0반음 / 포먼트 유지",
                              curve=curve))
    for ratio, amplitude, direction in ((.9, 2, -1), (.9, 2, 1),
                                        (1.1, 2, -1), (1.1, 2, 1),
                                        (.8, 3, -1), (1.25, 3, 1)):
        curve = [0., float(direction * amplitude), float(-direction * amplitude), 0.]
        tag = "down_up" if direction == -1 else "up_down"
        items.append(dict(name=f"combined_f{str(ratio).replace('.', 'p')}_p{amplitude}st_{tag}",
                          group="combined", ratio=ratio, curve=curve,
                          label=f"포먼트 {ratio:.2f}배 + 피치 0→{curve[1]:+g}→{curve[2]:+g}→0반음"))
    return items


def player(label, file):
    return (f'<div class="clip"><div>{html.escape(label)}</div>'
            f'<audio controls preload="none" src="{html.escape(file)}"></audio></div>')


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "reports/overnight-v1/protocol.json").read_text())
    frozen = {name: digest for name, digest in protocol["source_sha256"].items()
              if not name.startswith("/")}
    for name, digest in frozen.items():
        assert sha_file(ROOT / name) == digest, name
    previous = json.loads((ROOT / "reports/overnight-v1/listening/formants/manifest.json").read_text())
    sources = {row["speaker"]: row["source"] for row in previous}
    groups = {"formant": "포먼트만", "pitch": "파일 내 피치 변화만", "combined": "포먼트 + 피치"}
    manifest = dict(purpose="Exploratory listening candidates; not previous training draws or a new training run",
                    sample_rate=RATE, clip_seconds=1.5, added_noise=False,
                    speed_change=False, masking=False, waveform_order="pitch -> formant -> RMS match and peak limit",
                    pitch_anchor_times_seconds=[0., .5, 1., 1.5],
                    pitch_note="Linear interpolation between anchors; Rubber Band2 R2 with formant preservation and delay compensation",
                    formant_note="Praat Change gender; floor65 ceiling600, pitch median0, range1, duration1; fixed ratio per file",
                    comparison_note="Each montage: original then six variants in the listed order, separated by0.4s zeros",
                    original_ambient_noise_preserved=True,
                    previous_training_ranges=dict(pitch_internal_anchors_semitones=[-1., 1.], formant_ratio=[1/1.1, 1.1]),
                    clips=[], montages=[], sources=[])
    lines = ["# 포먼트·파일 내 피치 다양화 듣기", "",
             "추가 소음·속도 변형·마스킹 없이 화자2와3의 동일한 출발 원본을 비교한다. 원래 녹음의 주변음은 남아 있다.",
             "새 학습을 실행한 결과가 아니라 청취용 후보다. 이전 학습보다 강한 변형도 포함한다.",
             "각 그룹 이어듣기는 원본 → 아래 번호1~6 순서이며, 클립 사이 간격은0.4초다. 개별 파일은16kHz·1.5초 PCM16이다.",
             "피치는 파일 전체를 일괄 올리지 않고0/0.5/1.0/1.5초 앵커 사이를 선형으로 오르내린다. 포먼트 단독은 피치를 유지하고, 피치 단독은 포먼트 보존 옵션을 쓴다.",
             "포먼트 비율은 파일 내 고정이다. 결합은 피치 → 포먼트 순서다. 변형본은 원본 RMS에 맞춘 후 필요시 피크0.99로 제한했다.", ""]
    sections = []
    gap = np.zeros(6400, dtype=np.float32)
    for speaker in ("spk02", "spk03"):
        source = sources[speaker]
        original, rate = sf.read(ROOT / "datasets/task2/fixed-1p5-v1" / source, dtype="float32")
        assert rate == RATE and original.shape == (SAMPLES,)
        folder = REPORT / speaker
        folder.mkdir(exist_ok=True)
        manifest["sources"].append(dict(speaker=speaker, path=source,
                                         file_sha256=sha_file(ROOT / "datasets/task2/fixed-1p5-v1" / source)))
        record = save_wave(folder / "original.wav", original)
        manifest["clips"].append(dict(speaker=speaker, label="원본", group="original", **record))
        rows = []
        pitched = {}
        for recipe in recipes():
            curve = recipe["curve"]
            if curve is None:
                wave = original.copy()
            else:
                key = tuple(curve)
                if key not in pitched:
                    pitched[key] = pitch_wave(original, curve)
                wave = pitched[key].copy()
            if recipe["ratio"] != 1.:
                wave = shift_formants(wave, recipe["ratio"])
            wave, gain = level_match(wave, original)
            assert wave.shape == (SAMPLES,)
            record = save_wave(folder / f"{recipe['name']}.wav", wave)
            manifest["clips"].append(dict(speaker=speaker, source=source,
                                          formant_ratio=recipe["ratio"], pitch_anchors_semitones=curve,
                                          final_level_gain=gain, group=recipe["group"], label=recipe["label"], **record))
            rows.append((recipe, wave, record))
        lines += [f"## {speaker} · 출발", "", f"원본 `{source}` · [원본]({speaker}/original.wav) · [전체 이어듣기]({speaker}/all_comparison.wav)", ""]
        content = [f'<h2>{speaker} · 출발</h2><p>{html.escape(source)}</p>',
                   player("원본", f"{speaker}/original.wav"),
                   player("원본 → 전체18변형", f"{speaker}/all_comparison.wav")]
        for group, title in groups.items():
            selected = [row for row in rows if row[0]["group"] == group]
            chunks = [original]
            for _, wave, _ in selected:
                chunks.extend((gap, wave))
            result = save_wave(folder / f"{group}_comparison.wav", np.concatenate(chunks))
            manifest["montages"].append(dict(speaker=speaker, group=group,
                                              clip_files=[f"{speaker}/original.wav"] + [r[2]["file"] for r in selected], **result))
            lines += [f"### {title}", "", f"[원본 → 6종 이어듣기]({speaker}/{group}_comparison.wav)", ""]
            content.append(f'<h3>{title}</h3>')
            content.append(player("원본 → 아래1~6 순서", f"{speaker}/{group}_comparison.wav"))
            for index, (recipe, _, record) in enumerate(selected, 1):
                lines.append(f"{index}. [{recipe['label']}]({record['file']})")
                content.append(player(f"{index}. {recipe['label']}", record["file"]))
            lines.append("")
        chunks = [original]
        for _, wave, _ in rows:
            chunks.extend((gap, wave))
        result = save_wave(folder / "all_comparison.wav", np.concatenate(chunks))
        manifest["montages"].append(dict(speaker=speaker, group="all",
                                          clip_files=[f"{speaker}/original.wav"] + [r[2]["file"] for r in rows], **result))
        sections.append('<section>' + ''.join(content) + '</section>')
    for name, digest in frozen.items():
        assert sha_file(ROOT / name) == digest, name
    manifest["frozen_training_sources_unchanged"] = True
    (REPORT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (REPORT / "README.md").write_text("\n".join(lines).rstrip() + "\n")
    (REPORT / "index.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>포먼트·피치 다양화 듣기</title><style>body{font:16px system-ui;max-width:900px;margin:30px auto;padding:0 18px;line-height:1.6}audio{width:100%;max-width:650px}section{padding:20px 0;border-top:1px solid #ccc}.clip{margin:15px 0}h3{margin-top:35px}</style><h1>포먼트·파일 내 피치 다양화</h1><p>화자당 원본 + 18변형. 추가 소음·속도·마스킹 없이 비교합니다. 이전 학습보다 강한 청취용 후보를 포함합니다. 각 이어듣기는 원본 → 아래1~6 순서입니다.</p>' + ''.join(sections) + '</html>\n')
    with zipfile.ZipFile(REPORT / "samples.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(REPORT.rglob("*")):
            if path.is_file() and path.suffix in (".wav", ".md", ".json", ".html"):
                archive.write(path, path.relative_to(REPORT))
    print(json.dumps(dict(clips=len(manifest["clips"]), montages=len(manifest["montages"]),
                          report=str(REPORT), frozen_sources_unchanged=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
