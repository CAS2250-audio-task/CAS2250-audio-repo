"""Ordered CV screening, confirmation and feature re-comparison; test stays excluded."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from experiments.auxiliary.features import FeatureConfig, SETS
from src.config import load_config
from src.engine import file_hash, write_json
from .augment import prepare_banks, VARIANTS
from .data import Inputs, SPEAKERS, prepare
from .study import ROOT, condition, condition_id, run_condition


def summarize(rows):
    data = pd.DataFrame([dict(set_id=r["set_id"], speaker=r["validation_speaker"],
        seed=r["seed"], f1=r["validation"]["macro_f1"], accuracy=r["validation"]["accuracy"],
        train_f1=r["original_train"]["macro_f1"]) for r in rows])
    return dict(mean_f1=float(data.f1.mean()), accuracy=float(data.accuracy.mean()),
        worst_speaker_f1=float(data.groupby("speaker").f1.mean().min()),
        per_set={str(k):float(v) for k,v in data.groupby("set_id").f1.mean().items()},
        per_speaker={str(k):float(v) for k,v in data.groupby("speaker").f1.mean().items()},
        seed_std=float(data.groupby("seed").f1.mean().std(ddof=0)),
        mean_train_f1=float(data.train_f1.mean()), runs=len(rows), seeds=sorted(data.seed.unique().tolist()))


def improves(candidate, reference):
    return (candidate["mean_f1"] >= reference["mean_f1"] + .005
        and candidate["worst_speaker_f1"] >= reference["worst_speaker_f1"] - .01
        and all(candidate["per_set"][s] >= reference["per_set"][s] - .005 for s in reference["per_set"]))


def initialize(study, cfg):
    study.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(cfg.data_root) / "manifest.json").read_text())
    sources = {str(p.relative_to(ROOT)):file_hash(p) for folder in
        (ROOT / "experiments/auxiliary_cv", ROOT / "experiments/auxiliary", ROOT / "src") for p in folder.glob("*.py")}
    protocol = dict(version="speaker-auxiliary-cv-v1", data=manifest,
        config=cfg.to_dict(), speakers=list(SPEAKERS), test_excluded="spk01",
        features=FeatureConfig().to_dict(), sets=SETS, source_sha256=sources,
        checkpoint="maximum original validation macro F1; earliest epoch on tie",
        scaler="original fold train valid frames only; reused for train augmentation",
        augmentation=dict(application_probability_per_component=.5, fixed_wave_variants=VARIANTS,
            variant_bank_shared_across_models_and_training_seeds=True,
            pitch="time-varying 0/random[-1,1]/random[-1,1]/0 semitone anchors, formant-preserved Rubber Band 2.0 R2",
            speed="pitch-preserved 0.9..1.1; preserve entire previous reviewed interval; skip infeasible rates; center zero padding",
            noise="ESC-50 four categories, 16 training sources disjoint from prior val/test, SNR U[10,30] relative to current full-clip power including silence's own background",
            mask="20..50ms on spec and aux values+availability together; 4..8 mel bins on spec only; spec fill -2",
            combinations="independent Bernoulli application; waveform transform order pitch, speed, noise; recompute all features"),
        selection=dict(screen_seed=42, confirmation_seeds=[42,43,44],
            improvement_min_mean_macro_f1=.005, maximum_worst_speaker_regression=.01,
            maximum_either_set_regression=.005, maximum_individual_augmentation_confirmations=2,
            optional_freeze_lr_trigger="mean train minus validation F1 >= .08 and validation F1 < .95",
            final_auxiliary_choice="keep set0 unless confirmed mean CV F1 improves by >= .005 with worst speaker regression <= .01"),
        test_evaluation=False)
    path = study / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Frozen CV protocol or source code changed; preserve this study and use a new version")
    if not path.exists():
        write_json(path, protocol)
    return protocol


def report(study, destination, stage):
    destination.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads(p.read_text()) for p in sorted((study / "runs").glob("*/summary.json"))]
    rows = [dict(run=r["run"], condition_id=r["condition_id"], set_id=r["set_id"], seed=r["seed"],
        validation_speaker=r["validation_speaker"], val_macro_f1=r["validation"]["macro_f1"],
        val_accuracy=r["validation"]["accuracy"], train_macro_f1=r["original_train"]["macro_f1"],
        best_epoch=r["best_epoch"], training_seconds=r["training_seconds"],
        augmentation="+".join(r["condition"]["augmentation"]) or "none",
        weight_decay=r["condition"]["weight_decay"], dropout=r["condition"]["dropout"],
        freeze_front=r["condition"]["freeze_front"], backbone_lr_ratio=r["condition"]["backbone_lr_ratio"],
        backbone_initial_sha256=r["backbone_initial_sha256"]) for r in summaries]
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(destination / "all_runs.csv", index=False)
        common = frame[frame.set_id.isin(["0","4"])]
        common.groupby("condition_id").agg(mean_cv_f1=("val_macro_f1","mean"),
            accuracy=("val_accuracy","mean"), runs=("run","count"), seeds=("seed","nunique")).reset_index().to_csv(
                destination / "condition_summary.csv", index=False)
        frame.groupby(["condition_id","set_id","seed"]).agg(cv_f1=("val_macro_f1","mean"),
            cv_accuracy=("val_accuracy","mean"), worst_speaker_f1=("val_macro_f1","min")).reset_index().to_csv(
                destination / "seed_summary.csv", index=False)
    write_json(destination / "progress.json", dict(stage=stage, completed_runs=len(rows),
        test_evaluated=False, complete=stage == "complete"))
    for p in study.glob("*.json"):
        shutil.copyfile(p, destination / p.name)
    for run in (study / "runs").glob("*"):
        if (run / "summary.json").exists() and not (destination / "runs" / run.name).exists():
            shutil.copytree(run, destination / "runs" / run.name,
                            ignore=shutil.ignore_patterns("*.pt", "*.npz"))
    (destination / "README.md").write_text(
        "# 화자별 개발 CV 실험\n\n"
        f"현재 단계: **{stage}**, 완료 학습: **{len(rows)}회**.\n\n"
        "데이터는 real-context-1p5-v1, 개발 화자는 spk02~05입니다. 각 fold train 609개 / "
        "validation 203개이며 silence도 연결된 화자와 함께 이동합니다. spk01은 평가하지 않습니다.\n\n"
        "주 지표는 증강하지 않은 validation의 화자 4-fold 평균 macro F1입니다. fold마다 "
        "ImageNet 초기값에서 새 모델을 학습하고 scaler를 train에서만 적합합니다. 개발 CV로 "
        "조건과 epoch를 선택하므로 독립 최종 평가 점수로 해석하지 않습니다.\n\n"
        "all_runs.csv는 fold별 결과, seed_summary.csv는 세트·seed별 CV와 최저 화자 점수입니다. "
        "condition_summary.csv에는 seed 수가 다른 선별/확인 조건이 섞여 있으므로 seeds 열을 함께 봅니다.\n\n"
        "파형 증강은 클립당 고정 2개 변형을 준비하고, 매 학습 draw에서 각 정책을 확률 0.5로 "
        "독립 적용합니다. 검증에는 원본만 사용합니다. 이 첫 실험은 무한히 새로운 파형을 생성하는 "
        "온라인 증강의 성능을 평가한 것은 아닙니다.\n\n"
        "6개 피크 예외 파일의 원본 검수 구간 보존과 신호 수치는 확인했으며 사람의 청취 검수는 "
        "아직 수행하지 않았습니다. 자동 검증 결과를 청취 완료로 표현하지 않습니다.\n\n"
        "실행별 best checkpoint와 재개 checkpoint는 runs/auxiliary-cv-v1에 보관합니다. "
        "중단 후 같은 pipeline 명령을 실행하면 완료 fold를 건너뛰고 미완료 epoch부터 재개합니다.\n")


class Driver:
    def __init__(self, study, cfg, destination):
        self.study, self.cfg, self.destination = study, cfg, destination
        self.inputs = Inputs(study)

    def evaluate(self, settings, stage, seeds=(42,), sets=("0","4")):
        write_json(self.study / "active.json", dict(stage=stage, condition=settings,
            seeds=list(seeds), sets=list(sets), state="running"))
        prepare_banks(self.study, self.cfg, settings["augmentation"])
        rows = run_condition(self.study, self.cfg, self.inputs, settings, sets, seeds)
        summary = summarize(rows)
        print(f"STAGE {stage} {condition_id(settings)} seeds={seeds} F1={summary['mean_f1']:.4f} worst={summary['worst_speaker_f1']:.4f}", flush=True)
        report(self.study, self.destination, stage)
        return summary

    def freeze(self, name, value):
        path = self.study / f"{name}.json"
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Selection changed on resume: {name}")
        write_json(path, value)

    def choose_individuals(self, reference, candidates, stage, maximum):
        ref_screen = self.evaluate(reference, stage + "_reference_screen")
        screens = [(settings, self.evaluate(settings, stage + "_screen")) for settings in candidates]
        promising = sorted([(s,m) for s,m in screens if improves(m, ref_screen)],
                           key=lambda pair:pair[1]["mean_f1"], reverse=True)[:maximum]
        ref_full = self.evaluate(reference, stage + "_reference_confirm", (42,43,44))
        effective = []
        for settings, _ in promising:
            confirmed = self.evaluate(settings, stage + "_confirm", (42,43,44))
            if improves(confirmed, ref_full):
                effective.append((settings, confirmed))
        best = max([(reference,ref_full)] + effective, key=lambda pair:pair[1]["mean_f1"])
        self.freeze(stage + "_selection", dict(reference=reference, screens=[dict(condition=s,metrics=m) for s,m in screens],
            confirmed_effective=[dict(condition=s,metrics=m) for s,m in effective], selected=best[0], selected_metrics=best[1]))
        return best, effective

    def execute(self):
        base = condition()
        self.evaluate(base, "baseline_screen")
        (selected, current), effective = self.choose_individuals(base,
            [condition(augmentation=[p]) for p in ("pitch","speed","noise","mask")], "augmentation", 2)
        if len(effective) > 1:
            policies = sorted({p for settings,_ in effective for p in settings["augmentation"]})
            combined = condition(augmentation=policies)
            screen = self.evaluate(combined, "augmentation_combination_screen")
            if improves(screen, self.evaluate(selected, "augmentation_selected_screen")):
                confirmed = self.evaluate(combined, "augmentation_combination_confirm", (42,43,44))
                if improves(confirmed, current):
                    selected, current = combined, confirmed
        self.freeze("augmentation_final", dict(selected=selected, metrics=current))
        decay = dict(selected, weight_decay=1e-3)
        dropout = dict(selected, dropout=.2)
        (regularized, regular_metrics), effects = self.choose_individuals(selected,
            [decay, dropout], "regularization", 2)
        if len(effects) > 1:
            combined = dict(selected, weight_decay=1e-3, dropout=.2)
            if improves(self.evaluate(combined,"regularization_combination_screen"),
                        self.evaluate(regularized,"regularization_selected_screen")):
                confirmed = self.evaluate(combined,"regularization_combination_confirm",(42,43,44))
                if improves(confirmed,regular_metrics):
                    regularized, regular_metrics = combined, confirmed
        selected, current = regularized, regular_metrics
        optional = current["mean_train_f1"] - current["mean_f1"] >= .08 and current["mean_f1"] < .95
        self.freeze("optional_stage_decision", dict(run=optional, reference=selected, reference_metrics=current))
        if optional:
            (selected,current), _ = self.choose_individuals(selected,
                [dict(selected,freeze_front=True), dict(selected,backbone_lr_ratio=.1)], "freeze_lr", 1)
        self.freeze("common_training_selection",dict(selected=selected,metrics=current))
        rankings = []
        for set_id in range(8):
            score = self.evaluate(selected,"all_feature_sets_screen",sets=(str(set_id),))
            rankings.append((str(set_id), score))
        self.freeze("feature_screen", dict(ranking=[dict(set_id=s,metrics=m)
            for s,m in sorted(rankings,key=lambda pair:pair[1]["mean_f1"],reverse=True)]))
        finalists = sorted(rankings,key=lambda pair:pair[1]["mean_f1"],reverse=True)[:2]
        confirmed = []
        for set_id in sorted({"0","4"} | {s for s,_ in finalists}):
            result = self.evaluate(selected,"feature_sets_confirm",(42,43,44),(set_id,))
            confirmed.append((set_id,result))
        baseline = next(m for s,m in confirmed if s == "0")
        eligible = [(s,m) for s,m in confirmed if m["mean_f1"] >= baseline["mean_f1"] + .005
                    and m["worst_speaker_f1"] >= baseline["worst_speaker_f1"] - .01]
        chosen = max(eligible,key=lambda pair:pair[1]["mean_f1"])[0] if eligible else "0"
        self.freeze("final_selection",dict(selected_set=chosen,common_training=selected,
            confirmed_ranking=[dict(set_id=s,metrics=m) for s,m in sorted(confirmed,key=lambda pair:pair[1]["mean_f1"],reverse=True)],
            selected_without_test=True,test_evaluated=False))
        write_json(self.study / "active.json",dict(stage="complete",state="complete",test_evaluated=False))
        report(self.study,self.destination,"complete")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--study",default="runs/auxiliary-cv-v1")
    parser.add_argument("--config",default="configs/task2_real_context.json")
    parser.add_argument("--reports",default="reports/auxiliary-cv-v1")
    args = parser.parse_args()
    study = Path(args.study)
    study.mkdir(parents=True,exist_ok=True)
    with (study / "driver.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        cfg = load_config(args.config)
        initialize(study,cfg)
        prepare(study,cfg)
        try:
            Driver(study,cfg,Path(args.reports)).execute()
        except Exception as exc:
            write_json(study / "failure.json",dict(type=type(exc).__name__,message=str(exc)))
            raise


if __name__ == "__main__":
    main()
