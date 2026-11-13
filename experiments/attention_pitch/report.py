"""Small durable result tables and per-speaker learning curves."""
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/attention-pitch-v1"
REPORT = ROOT / "reports/attention-pitch-v1"
ORDER = ["baseline", "attention", "pitch", "attention_pitch", "compact8", "compact6"]


def render():
    REPORT.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text(encoding="utf-8")) for p in (REPORT / "folds").glob("*/*.json")]
    flat = [dict(candidate=r["candidate"], speaker=r["speaker"], best_epoch=r["best_epoch"],
                 final_epoch=r["final_epoch"], parameters=r["parameters"], **r["metrics"]) for r in records]
    summary = []
    if flat:
        frame = pd.DataFrame(flat)
        frame.to_csv(REPORT / "fold_results.csv", index=False)
        for name in ORDER:
            group = frame[frame.candidate.eq(name)]
            if len(group):
                summary.append(dict(candidate=name, folds=len(group), parameters=int(group.parameters.iloc[0]),
                    zero_f1=float(group.zero_f1.mean()), zero_accuracy=float(group.zero_accuracy.mean()),
                    worst_f1=float(group.zero_f1.min()), clean_f1=float(group.clean_f1.mean()),
                    clean_accuracy=float(group.clean_accuracy.mean()), noise_only_fpr=float(group.noise_only_fpr.mean())))
        pd.DataFrame(summary).to_csv(REPORT / "summary.csv", index=False)
    lines = ["# Attention·발화 구간 피치·경량화 실험", "",
             "기준은 이전 선택본: 기존 소음 p0.5·SNR[-5,20], 약한 포먼트 p0.5, Dropout0.2, WD1e-4, 보조없음·필터없음.",
             "4fold:02~05 각각 검증, 나머지3화자 학습. test01 음성은 읽거나 평가하지 않는다.",
             "동일 noise epoch10 LAST 모델·AdamW·RNG에서11~100epoch, batch16·609draw/epoch. encoder1e-5/head5e-4 유지, 전체 인코더 학습.",
             "마스킹은 실제 혼합 파형에서 고정 Silero가 산출한 확률만 사용한다. soft key bias log(g), speech-weighted mean, g=.05+.95p; 경계80ms 확장·무검출 시 전체 구간 유지.",
             "피치는 원본 자동VAD 구간 안의 PCHIP 곡선8종/파일, 랜덤 변곡 위치·방향·강도, 최대 변화속도32반음/초. 피치 변형 전후 발화 바깥은 동일하다. 기존 공통 포먼트는 이후 기존 방식 그대로 적용하며, 추가 소음은 항상 마지막에 혼합한다.",
             "원본VAD 구간을 평가에 주는 결과는 source-region 진단이며 배포 성능이나 후보 선택에 쓰지 않는다. 경계는 수동 정답이 아닌 기존 자동 검출 기록이다.",
             "기준12블록과 같은100epoch 예산으로 선택된 조건의8/6블록을 추가 비교한다. 균등한 깊이에서 남길 층을 고정하고 해당 파라미터·AdamW 상태를 보존한다.", "",
             "| 조건 | 완료fold | 파라미터 | 0dB F1 | 정확도 | 최저화자 F1 | 원본 F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['candidate']} | {row['folds']}/4 | {row['parameters']:,} | {row['zero_f1']*100:.2f}% | {row['zero_accuracy']*100:.2f}% | {row['worst_f1']*100:.2f}% | {row['clean_f1']*100:.2f}% |")
    lines += ["", "조건 채택: 기준보다4fold 평균 F1 +0.5%p 이상, 최저 개별 화자 하락1%p 이내. raw 최고점도 별도로 남긴다.",
              "경량화 허용: 선택12블록 대비 평균 F1 하락0.5%p 이내·개별 화자 하락1%p 이내. 통과 모델 중 가장 작은 모델을 표시한다.", ""]
    for name in ("selection.json", "compression_selection.json", "optional_skips.json"):
        path = REPORT / name
        if path.exists():
            lines += [f"`{name}`", "", "```json", path.read_text(encoding="utf-8").rstrip(), "```", ""]
    lines += ["[현재 진행](progress.json) · [고정 조건](protocol.json) · [검증 검사](preflight.json) · [진단](diagnostics.csv)",
              "", "![화자별 학습곡선](speaker_curves.png)", ""]
    (REPORT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    histories = []
    for name in ORDER:
        for path in (STUDY / "candidates" / name).glob("*/history.csv"):
            group = pd.read_csv(path)
            group["candidate"], group["speaker"] = name, path.parent.name
            histories.append(group)
    if histories:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        history = pd.concat(histories, ignore_index=True)
        history.to_csv(REPORT / "history.csv", index=False)
        fig, axes = plt.subplots(2, 4, figsize=(19, 7), sharex=True)
        for index, speaker in enumerate(("spk02", "spk03", "spk04", "spk05")):
            for name in ORDER:
                group = history[history.speaker.eq(speaker) & history.candidate.eq(name)]
                if len(group):
                    axes[0, index].plot(group.epoch, group.zero_f1 * 100, label=name, alpha=.85)
                    axes[1, index].plot(group.epoch, group.train_loss, label=name, alpha=.85)
            axes[0, index].set(title=speaker, ylabel="0dB monitor F1 (%)")
            axes[1, index].set(xlabel="epoch", ylabel="train loss")
            axes[0, index].grid(alpha=.2)
            axes[1, index].grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(REPORT / "speaker_curves.png", dpi=140)
        plt.close(fig)
