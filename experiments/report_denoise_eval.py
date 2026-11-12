"""Paired development-CV scores for the fixed denoising comparison."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import engine
from experiments.denoise_eval.filters import ROOT, STUDY, REPORT, FILTERS

NAMES = {"none": "필터 없음", "deepfilternet3": "DeepFilterNet3", "dns64": "Meta DNS64", "nsnet2": "Microsoft NSNet2"}


def main():
    p = pd.read_csv(REPORT / "predictions.csv.gz")
    keys = ["filter", "heldout_speaker", "kind", "noise_id", "category", "snr_db", "draw"]
    assert len(p) == 83760 and not p.duplicated(keys+["path"]).any()
    units = []
    for key, g in p.groupby(keys):
        assert len(g) == (20 if key[2] == "noise_only" else 203)
        words, silence = g[g.true_idx.ne(10)], g[g.true_idx.eq(10)]
        units.append(dict(zip(keys,key), **engine.metrics(g.true_idx, g.pred_idx, 11), n=len(g),
            keyword_accuracy=float(words.true_idx.eq(words.pred_idx).mean()) if len(words) else 0.,
            keyword_silence_rate=float(words.pred_idx.eq(10).mean()) if len(words) else 0.,
            false_command_rate=float(silence.pred_idx.ne(10).mean())))
    units = pd.DataFrame(units)
    assert len(units) == 528
    units.to_csv(REPORT / "condition_metrics.csv", index=False)
    clean = units[units.kind.eq("clean")].groupby("filter")[["accuracy", "macro_f1"]].mean()
    mixed = units[units.kind.eq("mixed")].groupby("filter")[["accuracy", "macro_f1", "keyword_accuracy", "keyword_silence_rate", "false_command_rate"]].mean()
    false = units[units.kind.eq("noise_only")].groupby("filter").false_command_rate.mean()
    noise = units[units.kind.eq("mixed")].groupby(["filter", "noise_id"])[["accuracy", "macro_f1"]].mean()
    fold = units[units.kind.eq("mixed")].groupby(["filter", "heldout_speaker"])[["accuracy", "macro_f1"]].mean()
    summary = mixed.rename(columns={"macro_f1":"mixed_macro_f1", "accuracy":"mixed_accuracy"}).join(
        clean.rename(columns={"accuracy":"original_accuracy", "macro_f1":"original_macro_f1"})).join(false.rename("noise_only_false_command_rate"))
    summary["delta_mixed_f1_pp"] = (summary.mixed_macro_f1-summary.loc["none", "mixed_macro_f1"])*100
    summary["delta_original_f1_pp"] = (summary.original_macro_f1-summary.loc["none", "original_macro_f1"])*100
    summary.to_csv(REPORT / "summary.csv")
    noise.to_csv(REPORT / "noise_summary.csv")
    fold.to_csv(REPORT / "speaker_summary.csv")
    assert abs(summary.loc["none", "mixed_macro_f1"]-.784748)<1e-6
    assert abs(summary.loc["none", "original_macro_f1"]-.992043)<1e-6
    pair_keys = ["heldout_speaker", "kind", "noise_id", "snr_db", "draw", "path", "true_idx"]
    baseline = p[p["filter"].eq("none")][pair_keys+["pred_idx"]]
    changes = []
    for name in FILTERS[1:]:
        paired = p[p["filter"].eq(name)].merge(baseline, on=pair_keys, suffixes=("", "_before"), validate="one_to_one")
        paired["before_correct"] = paired.true_idx.eq(paired.pred_idx_before)
        paired["after_correct"] = paired.true_idx.eq(paired.pred_idx)
        for kind, g in paired.groupby("kind"):
            changes.append(dict(filter=name,kind=kind,n=len(g),
                fixed=int((~g.before_correct & g.after_correct).sum()),
                broken=int((g.before_correct & ~g.after_correct).sum()),
                both_wrong=int((~g.before_correct & ~g.after_correct).sum())))
        paired[~paired.before_correct | ~paired.after_correct].to_csv(REPORT / f"{name}_paired_errors.csv.gz", index=False)
    pd.DataFrame(changes).to_csv(REPORT / "paired_changes.csv",index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12,4.6))
    plot_names = ["none", "deepfilternet3", "dns64", "nsnet2"]
    x = np.arange(4)
    axes[0].bar(x-.18, summary.loc[plot_names].original_macro_f1*100, .36, label="Original validation")
    axes[0].bar(x+.18, summary.loc[plot_names].mixed_macro_f1*100, .36, label="0 dB validation")
    axes[0].set(xticks=x,xticklabels=plot_names,ylim=(0,105),ylabel="Macro-F1 (%)",title="Noise-trained Conformer; four held-out-speaker folds")
    axes[0].legend()
    pivot = noise.macro_f1.unstack(0).loc[:,plot_names]*100
    pivot.plot.bar(ax=axes[1],width=.8)
    axes[1].set(ylabel="0 dB macro-F1 (%)",xlabel="Added-noise recording",ylim=(0,105),title="Same original mixture inputs for all filters")
    axes[1].tick_params(axis="x", rotation=35)
    for ax in axes:ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(REPORT/"denoising_comparison.png",dpi=170);plt.close(fig)
    latency = pd.read_csv(REPORT / "latency_summary.csv") if (REPORT/"latency_summary.csv").exists() else None
    lines = ["# 소음 혼합 Conformer: 잡음 제거기 3개 비교", "",
      "사용자 요청에 따라 소음 혼합으로 학습한 Conformer의 기존 4-fold 최고 checkpoint만 고정하고, "
      "추가 학습 없이 세 공개 잡음 제거기를 앞단에 적용했다. 무증강 학습본이나 속도·마스킹 결합 학습본은 비교하지 않았다.", "",
      "## 결과", "", "| 입력 필터 | 원본 F1 | 0dB F1 | 0dB 변화 | 잡음 단독 오반응률 |",
      "| --- | ---: | ---: | ---: | ---: |"]
    for name in FILTERS:
        r=summary.loc[name]
        lines.append(f"| {NAMES[name]} | {r.original_macro_f1*100:.2f}% | {r.mixed_macro_f1*100:.2f}% | {r.delta_mixed_f1_pp:+.2f}%p | {r.noise_only_false_command_rate*100:.2f}% |")
    best = summary.mixed_macro_f1.idxmax()
    lines += ["", f"이번 개발 검증에서 0dB 평균 F1이 가장 높은 조건은 **{NAMES[best]}**다. "
        f"필터 없음 대비 {summary.loc[best,'delta_mixed_f1_pp']:+.2f}%p이며, "
        f"원본 F1 변화는 {summary.loc[best,'delta_original_f1_pp']:+.2f}%p다. "
        "듣기에 잡음이 줄어드는 정도와 명령어 분류 성능을 구분해서 판단한다.", "",
        "이번 앞단 적용은 **0dB 개선과 원본 손상·잡음 오반응 사이의 교환**이 있다. "
        "잡음 단독 오반응률까지 함께 보고 실제로 적용할 조건을 판단해야 한다. "
        "현재 결과만으로 필터를 기본 경로에 자동 채택하지 않았다.", "",
        "![전체 및 잡음별 F1](denoising_comparison.png)", "",
        "점수는 4개 개발 화자의 held-out-speaker CV다. 0dB는 fold4×잡음8×추출3=96개 조건의 "
        "11-class macro-F1을 같은 가중치로 평균했다. 원본은 fold4 평균이다. "
        "원본812개, 0dB 혼합19,488개, 잡음 단독640개, 총20,940개에 필터 없음/3개 필터를 각각 평가했다. "
        "서로 다른 발화19,488개를 수집한 것이 아니라 원본812개를 소음·추출 조건별로 반복한 것이다.", "",
        "기존 혼합20,128개는 원래 평가의 SHA256과 모두 일치한다. 필터 없는20,940개 분류 결과도 이전 소음 혼합 학습본의 예측과 전부 일치했다. "
        "-5/-10/+10dB는 이번 추가 실험 범위에 포함하지 않았다.", "", "## 잡음별 0dB F1", "",
        "| 잡음 | 필터 없음 | DeepFilterNet3 | DNS64 | NSNet2 |", "| --- | ---: | ---: | ---: | ---: |"]
    for noise_id in pivot.index:
        lines.append("| "+noise_id+" | "+" | ".join(f"{pivot.loc[noise_id,name]:.2f}%" for name in FILTERS)+" |")
    lines += ["", "## 적용 방식", "",
        "- 세 필터 모두 혼합 파형만 입력받는다. 깨끗한 원본, 분리된 잡음, 정답, 소음 종류, SNR은 필터 입력에 없다.",
        "- DeepFilterNet3: 공식48kHz 모델.16k→48k→16k, 공식 sinc_fast/float32 리샘플링과 STFT 지연480샘플 보정. postfilter와 감쇠 제한 없음.",
        "- Meta DNS64: 공식16kHz Demucs(hidden64), dry=0. 모델 내부의 혼합 신호 기반 정규화와 복원을 그대로 사용한다.",
        "- Microsoft NSNet2: 공식16k ONNX, FFT320/홉160/sqrt-Hann, 최소 이득−80dB. 공식 복원 후 오른쪽 flush160샘플만 제외한다.",
        "- 모든 결과는16kHz·1.5초·float32. 외부 피크 정규화/클리핑/원본과의 혼합/원본에 맞춘 지연 탐색을 하지 않는다. 파일 간 모델/특징 상태를 초기화한다.",
        "- 분류기4개는 기존 clean-best epoch를 그대로 쓴다. 필터별 epoch/분류 임계값/혼합 비율을 고르거나 모델을 재학습하지 않았다.",
        "- 0dB는 원래1.5초 전체 파형 파워 대 추가 잡음 파워다. 원본에 이미 배경음/패딩이 있어 무잡음 활성 발화 기준0dB와 다르다. silence/잡음 단독은 각 fold 훈련 명령어600개의 RMS 중앙값 기준이다.", ""]
    if latency is not None:
        lines += ["## 추가 처리 시간", "", "RTX3090, 서버CPU2스레드, batch1, 1.5초 입력. 워밍업20회 후100회. 원본 CPU 파형부터 CPU 출력까지의 중앙값이다.", "",
            "| 필터 | 필터만 | 필터+분류 전체 | 전체 p95 |", "| --- | ---: | ---: | ---: |"]
        for name in FILTERS:
            a=latency[latency["filter"].eq(name)&latency.phase.eq("filter_only")].iloc[0]
            b=latency[latency["filter"].eq(name)&latency.phase.eq("filter_plus_classifier")].iloc[0]
            lines.append(f"| {NAMES[name]} | {a.median_ms:.2f}ms | {b.median_ms:.2f}ms | {b.p95_ms:.2f}ms |")
        lines += ["", "DFN3/DNS64의 신경망과 분류기는GPU, NSNet2는ONNX CPU를 사용했다. 왕복 리샘플·상태 초기화·전송·분류기Fbank를 포함한다. "
            "모델 로딩·WAV 디코딩·1.5초 창 확보·통신은 제외하며 스트리밍 알고리즘 지연 측정은 아니다. [측정 설정](latency_protocol.json)", ""]
    lines += ["## 근거와 한계", "",
        "- 개발 화자4명·훈련seed1개이며 같은 개발 CV에서 필터를 고르는 결과다. 새 화자/녹음환경/최종 test 일반화를 확정하지 않는다. 이번 실행은 test spk01 오디오를 읽거나 추론하지 않았다. 프로젝트 과거 브랜치의 test 사용 이력과 별개다.",
        "- 평가 잡음8개는 우리 분류기 훈련 잡음6개와 원본 단위로 분리했다. 외부 잡음 제거기의 DNS 사전학습 전체 원본과 평가 잡음의 중복은 완전히 감사하지 못했으므로, 잡음 제거기에도 미관측 잡음이라고 주장하지 않는다.",
        "- 이미 소음으로 학습한 KWS는 범용 음질 개선기를 거쳐도 정확도가 자동으로 높아지지 않는다. 발화 특징 손실과 분류기 입력 분포 변화도 같이 측정해야 한다. [관련 KWS 연구](https://arxiv.org/abs/2309.16060)",
        "- 공식 출처: [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), [Meta denoiser](https://github.com/facebookresearch/denoiser), "
        "[Microsoft NSNet2](https://github.com/microsoft/DNS-Challenge/tree/a052ad5a7714bcc6069b515666e837bf973099de/NSNet2-baseline). 정확한 revision/가중치/코드 SHA는 [고정 프로토콜](protocol.json)에 있다.",
        "", "검증: [독립 재검사](independent_audit.json), [필터별 요약](summary.csv), [화자별 결과](speaker_summary.csv), "
        "[동일 파일에서 고쳐진/망가진 개수](paired_changes.csv), [듣기 샘플](../../runs/denoise-eval-v1/listening/README.md).", ""]
    (REPORT/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(summary.to_string(),flush=True)


if __name__ == "__main__":
    main()
