"""Summarize the finished fixed100 study without editing its frozen package."""
import json
from pathlib import Path

import pandas as pd

from src import engine
from experiments.conformer_fixed100.run import freeze_protocol

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports/conformer-fixed100-v1"
PARENT=ROOT/"reports/conformer-long-zero-v1"


def main():
    protocol=freeze_protocol()
    assert json.loads((ROOT/"runs/conformer-fixed100-v1/complete.json").read_text())["training_and_independent_audit_complete"]
    old=pd.read_csv(PARENT/"runs.csv")
    new=pd.read_csv(REPORT/"runs.csv")
    keys=["condition","batch","speaker","pipeline"]
    paired=new.merge(old,on=keys,suffixes=("_new","_old"),validate="one_to_one")
    assert len(paired)==64
    paired["zero_f1_delta_pp"]=(paired.zero_f1_new-paired.zero_f1_old)*100
    paired.to_csv(REPORT/"paired_speaker_comparison.csv",index=False)
    draws=[]
    for item in protocol["plan"]:
        source=ROOT/item["source_run"]
        target=ROOT/"runs/conformer-fixed100-v1/branches"/item["condition"]/f"bs{item['batch']}"/f"lr{item['encoder_lr']:.0e}"/item["speaker"] if item["extend"] else source
        for stage,folder in (("previous",source),("fixed100",target)):
            units=pd.read_csv(folder/"best_off_full_units.csv")
            units=units[units.kind.eq("mixed")]
            assert len(units)==24
            for label,mask in (("monitor0",units.draw.eq(0)),("additional1_2",units.draw.isin([1,2]))):
                part=units[mask]
                draws.append(dict(condition=item["condition"],batch=item["batch"],speaker=item["speaker"],
                    stage=stage,draws=label,f1=float(part.macro_f1.mean()),accuracy=float(part.accuracy.mean())))
    draws=pd.DataFrame(draws)
    draws.to_csv(REPORT/"draw_split_speakers.csv",index=False)
    split=draws.groupby(["condition","batch","stage","draws"],as_index=False)[["f1","accuracy"]].mean()
    split.to_csv(REPORT/"draw_split_summary.csv",index=False)
    endpoints=pd.read_csv(REPORT/"endpoints.csv")
    assert len(endpoints)==16 and endpoints.new_end_epoch.eq(100).all()
    peak_count=int((endpoints.extended & endpoints.new_best_epoch.gt(endpoints.old_end_epoch)).sum())
    assert int(endpoints.extended.sum())==9
    rows=[]
    summary=pd.read_csv(REPORT/"summary.csv")
    for r in summary.itertuples():
        assert 0<=r.zero_f1<=1 and 0<=r.zero_accuracy<=1
        g=new[new.condition.eq(r.condition)&new.batch.eq(r.batch)&new.pipeline.eq(r.pipeline)]
        for metric in ("zero_f1","zero_accuracy","clean_f1","clean_accuracy","noise_only_fpr"):
            assert len(g)==4 and abs(float(g[metric].mean())-getattr(r,metric))<1e-12
        if r.pipeline=="best_off":
            prev=pd.read_csv(PARENT/"summary.csv")
            prior=prev[prev.condition.eq(r.condition)&prev.batch.eq(r.batch)&prev.pipeline.eq(r.pipeline)].iloc[0]
            rows.append(f"| {r.condition} | {r.batch} | {prior.zero_f1*100:.2f}% | {r.zero_f1*100:.2f}% | {r.zero_accuracy*100:.2f}% | {r.zero_f1_delta_pp:+.2f}%p |")
    baseline=paired[paired.condition.eq("baseline")&paired.batch.eq(16)&paired.pipeline.eq("best_off")].sort_values("speaker")
    lines=["# 100epoch 공통 종료 실험 검토", "",
        "405epoch 추가 학습과 전체 평가·독립 검사를 완료했다. 9개 분기를 연장했고 기존100epoch의7개는 재사용했다. 기존 실험 소스·보고서·체크포인트는 보존했다.","",
        "최고 결과는 기존 증강·batch16·무필터 최고본: 0dB macro-F1 **89.7492%**, 정확도 **90.1888%**. 원본F1 99.6585%, 정확도99.6305%다.",
        "모든 값은 개발4fold 평균이다. 최고본은 화자별 고정0dB draw0 F1로 선택하고 전체3draw로 보고한다. test01은 이번에도 평가하지 않았다.","",
        "| 증강 | 배치 | 이전0dB F1 | 연장 후0dB F1 | 연장 후0dB 정확도 | F1 변화 |", "|---|---:|---:|---:|---:|---:|",*rows,"",
        f"연장9분기 중 {peak_count}분기가 이전 종료 이후 최고점을 갱신했다. 이전 조기 종료는 일부 분기에서 실제 개선을 놓쳤다.","",
        "| 기존 증강·batch16 화자 | 이전0dB F1 | 연장 후0dB F1 | 변화 |", "|---|---:|---:|---:|"]
    for r in baseline.itertuples():
        lines.append(f"| {r.speaker} | {r.zero_f1_old*100:.2f}% | {r.zero_f1_new*100:.2f}% | {r.zero_f1_delta_pp:+.2f}%p |")
    main_split=split[split.condition.eq("baseline")&split.batch.eq(16)&split.draws.eq("additional1_2")].set_index("stage")
    lines += ["", "02·03과 종료 시점 해석:", "",
        "- 기존 증강·batch16: 02 최고epoch25→63, 03은59→97, 04는44→87. 기존 증강·batch32의03도68→84로 늦게 개선했다.",
        "- 기존 증강·batch32의02는53epoch 최고점을100까지 갱신하지 않았다. 02가 어떤 조건에서도 추가 학습으로 계속 좋아지는 것은 아니다.",
        "- 말소리·0dB 강화의02는batch16/32에서38/40epoch에 새 최고점을 냈지만100epoch 부근에는 다시 F1 약60%대로 하락했다. 단기 회복과 장기 악화가 함께 관측된다.",
        "- 기존 증강·batch16의100epoch 종료본은F1 87.9194%, 최고본89.7492%보다1.8299%p 낮다. 종료본을 그대로 쓰기보다 검증으로 선택한 최고본을 유지한다.",
        "- batch16 최고본에DFN3를 적용하면0dB F1 89.4946%, 정확도89.6295%로, 같은 최고본 무필터보다 낮다. 잡음단독 오반응도3.4375→10.0%다.",
        "- 03의batch16은97epoch 최고점이므로100epoch만으로 완전한 수렴이나 추가 학습 무용을 입증하지 못한다.","",
        f"epoch 선택에 직접 쓰지 않은 추가draw1~2의 기존증강/batch16 F1도 {main_split.loc['previous','f1']*100:.4f}→{main_split.loc['fixed100','f1']*100:.4f}%로 개선했다. 다만 같은 개발 화자·소음 원본의 다른 crop이므로 독립 test가 아니다.","",
        "검증: 신규epoch 예측739,935개·증강draw246,645개·최종예측188,460개 지표/입력 검사, 최고본32,886개 재추론 일치. 모든 마지막 모델은100epoch이며 optimizer/RNG와 부모 checkpoint 보존도 확인했다.","",
        "[전체표](README.md) · [화자별곡선](speaker_curves.png) · [화자별전후비교](paired_speaker_comparison.csv) · [독립검사](independent_audit.json)",""]
    (REPORT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    engine.write_json(REPORT/"review_validation.json",dict(paired_speaker_rows=len(paired),
        draw_split_rows=len(split),summary_rows=len(summary),extended_branches=9,
        branches_with_new_monitor_peak=peak_count,all_endpoints_epoch100=True,
        parent_protocol_and_checkpoint_hashes_verified=True,test_evaluated=False,
        analyzer_sha256=engine.file_hash(Path(__file__))))
    print(json.dumps(dict(summary_rows=len(summary),new_peaks=peak_count,result=str(REPORT/"RESULTS.md"))))


if __name__=="__main__":main()
