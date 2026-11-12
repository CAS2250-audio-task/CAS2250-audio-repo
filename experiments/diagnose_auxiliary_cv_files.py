"""Recover epoch predictions for a fixed CV model without changing the study."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.auxiliary.data import audit_dataset, make_jobs
from experiments.auxiliary.features import encode
from experiments.auxiliary.model import state_hash
from experiments.auxiliary.study import evaluate
from experiments.auxiliary_cv.data import Inputs, original_cache
from experiments.auxiliary_cv import study as cv
from src.config import load_config
from src.engine import file_hash, write_json

SOURCE = Path("runs/auxiliary-cv-v1")
DEST = Path("runs/auxiliary-cv-file-diagnostics-v1")
REPORT = Path("reports/auxiliary-cv-v1/file_diagnostics")
SEED = 44
SET_ID = "3"
EXTERNAL_CACHE = Path("runs/auxiliary-cv-file-diagnostics-v1/cache/spk01")


def save_predictions(path, predictions, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = predictions.copy()
    frame["true_label"] = frame.true_idx.map(dict(enumerate(labels)))
    frame["pred_label"] = frame.pred_idx.map(dict(enumerate(labels)))
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def replay(inputs, settings, speaker):
    cfg = load_config("configs/task2_real_context.json")
    name = f"c{cv.condition_id(settings)}_set{SET_ID}_seed{SEED}_{speaker}"
    original, run = SOURCE / "runs" / name, DEST / "runs" / name
    output = REPORT / name
    output.mkdir(parents=True, exist_ok=True)
    saved_evaluate, saved_atomic = cv.evaluate, cv.atomic_save
    current = 0
    if (run / "last.pt").exists():
        current = int(torch.load(run / "last.pt", map_location="cpu", weights_only=False)["epoch"])

    def capture_evaluate(model, loader, device, labels):
        nonlocal current
        scores, predictions = saved_evaluate(model, loader, device, labels)
        if len(predictions) == 203:
            current += 1
            if current <= 10:
                save_predictions(output / f"epoch_{current:02d}.csv", predictions, labels)
                write_json(output / f"epoch_{current:02d}_metrics.json", scores)
        return scores, predictions

    def capture_save(value, path):
        saved_atomic(value, path)
        if speaker == "spk02" and Path(path).name == "last.pt":
            snapshots = run / "epoch_models"
            snapshots.mkdir(exist_ok=True)
            saved_atomic(dict(epoch=value["epoch"], model={k:v.detach().cpu()
                for k,v in value["model"].items()}), snapshots / f"epoch_{value['epoch']:02d}.pt")

    cv.evaluate, cv.atomic_save = capture_evaluate, capture_save
    try:
        result = cv.run_fold(DEST, cfg, inputs, settings, SET_ID, SEED, speaker)
    finally:
        cv.evaluate, cv.atomic_save = saved_evaluate, saved_atomic
    old_state = torch.load(original / "best.pt", map_location="cpu", weights_only=True)
    new_state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    old_summary = json.loads((original / "summary.json").read_text())
    checks = dict(speaker=speaker, original_run=name,
        history_csv_identical=file_hash(original / "history.csv") == file_hash(run / "history.csv"),
        selected_model_tensors_identical=state_hash(old_state["model"]) == state_hash(new_state["model"]),
        selected_predictions_identical=file_hash(original / "validation/predictions.csv") == file_hash(run / "validation/predictions.csv"),
        scores_identical=result["validation"] == old_summary["validation"],
        epoch_predictions_complete=all((output / f"epoch_{e:02d}.csv").exists() for e in range(1,11)))
    write_json(output / "replay_audit.json", checks)
    assert all(v for k,v in checks.items() if k not in ("speaker", "original_run")), checks
    print("REPLAY_VERIFIED", json.dumps(checks, ensure_ascii=False), flush=True)


class ExternalClips(Dataset):
    def __init__(self, frame, spec, auxiliary, labels):
        self.frame, self.spec, self.auxiliary = frame, spec, auxiliary
        self.targets = frame.label.map({label:i for i,label in enumerate(labels)}).to_numpy()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return (torch.from_numpy(self.spec[index]), torch.from_numpy(self.auxiliary[index]),
                int(self.targets[index]), self.frame.path.iloc[index])


def diagnose_spk01(settings):
    cfg = load_config("configs/task2_real_context.json")
    cfg.seed = SEED
    frame, _ = audit_dataset(cfg)
    frame = frame[frame.split == "test"].reset_index(drop=True)
    assert len(frame) == 203 and frame.group.eq("speaker:spk01").all()
    directory = EXTERNAL_CACHE
    original_cache(directory, frame, make_jobs(frame, cfg), cfg, workers=8)
    with np.load(directory / "inputs.npz", allow_pickle=False) as bank:
        spec, raw, valid = [bank[k].copy() for k in ("spec", "raw", "valid")]
        assert np.array_equal(bank["paths"], frame.path.to_numpy())
    scaler = json.loads((SOURCE / "folds/spk02/scaler.json").read_text())
    auxiliary = encode(raw, valid, scaler, SET_ID)
    dataset = ExternalClips(frame, spec, auxiliary, cfg.labels)
    loader = cv.batches(dataset, cfg)
    model = cv.CVModel(auxiliary.shape[1], pretrained=False).cuda()
    name = f"c{cv.condition_id(settings)}_set{SET_ID}_seed{SEED}_spk02"
    output = REPORT / "spk01_external_model_spk02_seed44"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch in range(1,11):
        state = torch.load(DEST / "runs" / name / "epoch_models" / f"epoch_{epoch:02d}.pt",
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        scores, predictions = evaluate(model, loader, torch.device("cuda"), cfg.labels)
        save_predictions(output / f"epoch_{epoch:02d}.csv", predictions, cfg.labels)
        write_json(output / f"epoch_{epoch:02d}_metrics.json", scores)
        rows.append(dict(epoch=epoch, **scores))
    pd.DataFrame(rows).to_csv(output / "history.csv", index=False)
    write_json(output / "protocol.json", dict(evaluation_speaker="spk01", source_model=name,
        source_training_speakers=["spk03", "spk04", "spk05"], fixed_comparison_epochs=[3,10],
        user_requested_post_selection_diagnostic=True, test_evaluated=True,
        no_training_or_scaler_fit_on_spk01=True, original_study_selection_unchanged=True,
        original_scaler_sha256=file_hash(SOURCE / "folds/spk02/scaler.json")))
    print("EXTERNAL_SPK01_COMPLETE", flush=True)


def main():
    global DEST, REPORT, SET_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-id", choices=("0", "3"), default="3")
    args = parser.parse_args()
    SET_ID = args.set_id
    if SET_ID == "0":
        DEST = Path("runs/auxiliary-cv-file-diagnostics-set0-v1")
        REPORT = Path("reports/auxiliary-cv-v1/file_diagnostics_set0")
    torch.set_num_threads(2)
    DEST.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    assert all(file_hash(path) == digest for path,digest in protocol["source_sha256"].items())
    for path in [SOURCE / "protocol.json", *sorted((SOURCE / "folds").glob("*/*.json"))]:
        target = DEST / path.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert file_hash(target) == file_hash(path)
        else:
            shutil.copyfile(path, target)
    selection = json.loads((SOURCE / "final_selection.json").read_text())
    assert selection["selected_set"] == "3"
    settings = selection["common_training"]
    write_json(REPORT / "scope.json", dict(seed=SEED, set_id=SET_ID, condition=settings,
        original_study="auxiliary-cv-v1", replay_speakers=["spk02", "spk03", "spk04"],
        spk01_separate_post_selection_evaluation_requested=True,
        purpose="recover per-file epoch transitions, not repeat model selection"))
    inputs = Inputs(SOURCE)
    for speaker in ("spk02", "spk03", "spk04"):
        replay(inputs, settings, speaker)
    del inputs
    diagnose_spk01(settings)
    write_json(REPORT / "complete.json", dict(complete=True, diagnostic_replay_epochs=30,
        original_study_formal_runs_unchanged=184, spk01_post_selection_diagnostic=True))


if __name__ == "__main__":
    main()
