"""Cross-check final comparison tables and local report links, then fingerprint them."""
import json
from pathlib import Path
import re
import time

import pandas as pd

from src import engine
from experiments.conformer_long_zero.run import ROOT,STUDY,REPORT,write_json


def near(a,b):
    assert abs(float(a)-float(b))<1e-10,(a,b)


def check():
    assert (STUDY/"complete.json").exists()
    audit=json.loads((REPORT/"independent_audit.json").read_text())
    tests=json.loads((REPORT/"regression_tests.json").read_text())
    assert tests["exit_code"]==0 and tests["passed"]==58
    assert audit["optimizer_update_counters_and_resume_states_verified"]
    compare=pd.read_csv(REPORT/"comparison.csv")
    split=pd.read_csv(REPORT/"draw_split_summary.csv")
    summary=pd.read_csv(REPORT/"summary.csv")
    prior=pd.read_csv(ROOT/"reports/conformer-noise-focus-v1/snr_summary.csv")
    assert len(compare)==24 and len(split)==48 and len(summary)==16
    for row in compare.itertuples():
        if row.stage.startswith("long_"):
            pipeline=row.stage.removeprefix("long_")+"_"+row.denoiser
            target=summary[summary.condition.eq(row.condition)&summary.batch.eq(row.batch)&summary.pipeline.eq(pipeline)].iloc[0]
            for field in ("zero_f1","clean_f1","noise_only_fpr"):near(getattr(row,field),target[field])
        if row.stage=="previous_clean_best":
            target=prior[prior.training_condition.eq(row.condition)&prior.denoiser.eq(row.denoiser)&prior.snr_db.eq(0)].iloc[0]
            near(row.zero_f1,target.macro_f1)
        subset=split[split.condition.eq(row.condition)&split.stage.eq(row.stage)&split.batch.eq(row.batch)&split.denoiser.eq(row.denoiser)].set_index("draw_subset")
        assert len(subset)==2 and subset.loc["monitor_draw0","metric_units"]==32 and subset.loc["additional_draw1_2","metric_units"]==64
        near(row.zero_f1,(subset.loc["monitor_draw0","zero_f1"]+2*subset.loc["additional_draw1_2","zero_f1"])/3)
        near(row.keyword_accuracy+row.keyword_silence_miss+row.keyword_other_wrong,1)
        if row.stage=="long_best" and row.denoiser=="off":
            targets=[]
            for speaker in ("spk02","spk03","spk04","spk05"):
                run=json.loads((REPORT/"runs"/f"{row.condition}_bs{row.batch}_lr1e-05_{speaker}.json").read_text())
                targets.append(run["best_monitor_f1"])
            near(sum(targets)/4,subset.loc["monitor_draw0","zero_f1"])
    errors=pd.read_csv(REPORT/"paired_error_changes.csv")
    assert len(errors)==8
    for row in errors.itertuples():
        assert row.wrong_to_correct+row.correct_to_wrong+row.both_wrong+row.both_correct==row.number_of_mixed_cases==19488
    common=pd.read_csv(REPORT/"common_epoch_summary.csv")
    means=pd.read_csv(REPORT/"learning_curve_means.csv")
    assert len(common)==20
    for row in common.itertuples():
        target=means[means.condition.eq(row.condition)&means.batch.eq(row.batch)&means.epoch.eq(row.epoch)].iloc[0]
        near(row.zero_f1,target.zero_f1);near(row.clean_f1,target.clean_f1)
    links=[]
    for name in ("README.md","RESULTS.md","VALIDATION.md"):
        path=REPORT/name
        for target in re.findall(r"\]\(([^)]+)\)",path.read_text()):
            if re.match(r"^[a-zA-Z]+:",target) or target.startswith("#"):continue
            resolved=(path.parent/target.split("#")[0]).resolve()
            assert resolved.exists(),(name,target)
            links.append(dict(document=name,target=target))
    result=dict(comparison_rows_cross_checked=24,draw_split_rows_reconciled=48,
        paired_mixed_error_tables_checked=8,common_epoch_groups_checked=20,
        local_document_links_checked=len(links),regression_tests_passed=tests["passed"],
        original_primary_results_match=True,monitor_and_all3draw_metrics_reconcile=True,
        analysis_sha256=engine.file_hash(ROOT/"experiments/analyze_conformer_long_zero.py"),
        checker_sha256=engine.file_hash(Path(__file__)),test_evaluated=False)
    progress=json.loads((REPORT/"progress.json").read_text())
    progress.update(stage="complete",analysis_complete=True,independent_audit_complete=True,
        updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),test_evaluated=False)
    write_json(REPORT/"progress.json",progress)
    write_json(STUDY/"progress.json",progress)
    write_json(REPORT/"analysis_validation.json",result)
    fingerprints={str(p.relative_to(REPORT)):engine.file_hash(p) for p in REPORT.rglob("*")
                  if p.is_file() and p.name!="artifact_checksums.json" and ".partial" not in p.name}
    write_json(REPORT/"artifact_checksums.json",fingerprints)
    write_json(STUDY/"postprocess_complete.json",dict(final_analysis_and_validation_complete=True,
        final_report_sha256=engine.file_hash(REPORT/"RESULTS.md"),artifact_count=len(fingerprints),test_evaluated=False))
    print(json.dumps(result))


if __name__=="__main__":check()
