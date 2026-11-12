"""Refresh a readable progress page and completed fold comparisons."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/"runs/overnight-v1"
REPORT=ROOT/"reports/overnight-v1"


def render():
    REPORT.mkdir(parents=True,exist_ok=True)
    p=json.loads((REPORT/"progress.json").read_text()) if (REPORT/"progress.json").exists() else {}
    lines=["# Overnight: 정규화 → 포먼트 증강 → 보조음향 → DFN3", "",
        f"상태: `{p.get('stage','preparing')}`, 조건 `{p.get('candidate','')}`, 화자 `{p.get('speaker','')}`, epoch `{p.get('epoch','')}`.","",
        "별도 브랜치 overnight-regularization-formant-aux. 원본 test01은 사용하지 않는다.",
        "공통 epoch10 LAST에서 모델·AdamW·RNG를 복원하고100까지 학습한다. 초기10epoch는 이전실험 공통 상태이며 새 후보 변화는epoch11부터 적용한다.",
        "4fold·seed42·batch16·609draw/epoch. 인코더1e-5/head5e-4 유지, 전체 인코더 학습, 조기 종료 없음.",
        "WD1e-3과encoder dropout0.2를 각각 비교, 둘 다 유효하면 결합. 선택 조건에서 A=SpecAugment+속도+피치, B=포먼트 한 설정, A+B를 비교한다. 기존소음혼합은공통으로유지한다.",
        "피치는파일내±1반음 anchor변화, 속도0.9~1.1배와발화보호, 포먼트비율1/1.1~1.1. 각정책p0.5독립이며7가지파형조합별고정2변형을준비한다. 실제혼합파형에서보조를추출하고시간마스킹은보조값·availability에도적용한다.",
        "보조는세트0·3·7로비교한다. 원본train 유효프레임으로만 scaler를 적합한다. aux CNN을 시간 정렬해인코더 출력에잔차로더하며256→11head유지, 초기aux projection은0이다.",
        "조건 선택은 전체3draw0dB 개발4fold F1이 기준보다0.5%p 이상 높고 최저화자F1하락1%p이내일때. 미세한변화는 통계적우월성으로주장하지않는다.",
        "마지막에는 선택된동일모델에DFN3off/on. 필터후파형에서보조도다시추출한다. 기존 공식B16필터캐시를재사용한다.",
        "저장: 각후보best와전체epoch기록보존. 완료분기의큰LAST optimizer는검증후삭제하고미완료분기만재개상태유지. 기존실험파일은수정하지않는다.","",
        "[실시간 상태](progress.json) · [로그](../../runs/overnight-v1/execution.log) · [프로토콜](protocol.json)",""]
    rows=[]
    for path in sorted((REPORT/"folds").glob("*/*.json")):
        s=json.loads(path.read_text());rows.append(dict(candidate=s["candidate"],speaker=s["speaker"],best_epoch=s["best_epoch"],**s["metrics"]))
    if rows:
        d=pd.DataFrame(rows);d.to_csv(REPORT/"fold_results.csv",index=False)
        summary=[]
        lines += ["| 후보 | 완료fold | 0dB F1 | 0dB 정확도 | 최저화자 F1 |", "|---|---:|---:|---:|---:|"]
        for name,g in d.groupby("candidate"):
            if len(g)!=4:
                lines.append(f"| {name} | {len(g)}/4 | 진행중 | | |")
                continue
            m=dict(candidate=name,zero_f1=float(g.zero_f1.mean()),zero_accuracy=float(g.zero_accuracy.mean()),
                worst_f1=float(g.zero_f1.min()),clean_f1=float(g.clean_f1.mean()),clean_accuracy=float(g.clean_accuracy.mean()),noise_only_fpr=float(g.noise_only_fpr.mean()))
            summary.append(m)
            lines.append(f"| {name} | 4/4 | {m['zero_f1']*100:.2f}% | {m['zero_accuracy']*100:.2f}% | {m['worst_f1']*100:.2f}% |")
        pd.DataFrame(summary).to_csv(REPORT/"summary.csv",index=False)
    histories=[]
    for path in sorted((STUDY/"candidates").glob("*/*/history.csv")):
        h=pd.read_csv(path);h["candidate"]=path.parent.parent.name;h["speaker"]=path.parent.name;histories.append(h)
    if histories:
        all_history=pd.concat(histories,ignore_index=True)
        all_history.to_csv(REPORT/"history.csv",index=False)
        fig,axes=plt.subplots(2,4,figsize=(18,8),sharex=True)
        for j,speaker in enumerate(("spk02","spk03","spk04","spk05")):
            for name,g in all_history[all_history.speaker.eq(speaker)].groupby("candidate"):
                axes[0,j].plot(g.epoch,g.zero_f1*100,label=name)
                axes[1,j].plot(g.epoch,g.zero_loss,label=name)
            axes[0,j].set(title=speaker,ylabel="0dB F1 draw0 (%)")
            axes[1,j].set(xlabel="epoch",ylabel="0dB validation loss")
        axes[0,0].legend(fontsize=6)
        fig.tight_layout();fig.savefig(REPORT/"speaker_curves.png",dpi=130);plt.close(fig)
        lines += ["", "![화자별 학습 진행](speaker_curves.png)",""]
    for stage in ("regularization","augmentation","auxiliary"):
        path=REPORT/f"selection_{stage}.json"
        if path.exists():lines += [f"선택 {stage}: `{json.loads(path.read_text())['selected']}`",""]
    if (REPORT/"independent_audit.json").exists():lines += ["[완료 검증](independent_audit.json)",""]
    (REPORT/"README.md").write_text("\n".join(lines)+"\n")
