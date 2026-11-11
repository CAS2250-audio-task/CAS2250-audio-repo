"""Reproducible set 0..7 training, stress evaluation, and frozen final test."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

from src.config import load_config
from src.engine import (choose_device, file_hash, git_info, metrics, save_reports,
                        seed_all, write_json)
from .features import FeatureConfig, NAMES, SETS, encode
from .data import NOISE_CATEGORIES, SNRS, prepare
from .model import AcousticModel, state_hash

ROOT = Path(__file__).resolve().parents[2]


def protocol(cfg):
    return dict(version="auxiliary-midfusion-v1", baseline_config=cfg.to_dict(),
                feature_config=FeatureConfig().to_dict(), feature_sets=SETS,
                representations=dict(log_f0="12*log2(Hz/100)", formants="Hz, train-valid-frame zscore",
                                     bandwidths="Hz, train-valid-frame zscore",
                                     derivatives="5-frame local slope per second, no missing gaps",
                                     log_rms="20*log10(max(RMS,1e-8))",
                                     flux="RMS difference of L1-normalized magnitude spectra",
                                     flags="availability only, not correctness; probabilities unchanged"),
                architecture=dict(backbone="reference one-channel ImageNet ResNet18",
                                  fusion="after layer1; concat + identity-initialized 1x1 projection",
                                  auxiliary="Conv1d 32/64/64, k5 d1/2/4, GroupNorm, GELU, dropout .2",
                                  time_alignment="input centres 4*j after ResNet stem/layer1",
                                  tail="unchanged layer2/3/4 -> avgpool -> 512 -> 11",
                                  freeze_backbone=False),
                training_noise_augmentation=False, training_pitch_augmentation=False,
                noise_categories=NOISE_CATEGORIES, added_snr_db=SNRS,
                snr_definition="original full-clip power, including existing noise, versus added noise; silence uses train-keyword median RMS",
                peak_control="scale mixed signal uniformly only if peak exceeds .99; preserves mixture SNR",
                checkpoint_selection="original validation macro F1; earliest epoch on a tie",
                screening_seed=42, confirmation_seeds=[42, 43, 44],
                finalist_selection="highest original validation macro F1 in screening, tie by mean noisy macro F1; keep baseline plus best two auxiliary sets",
                final_model_selection="highest mean original validation macro F1 over seeds 42/43/44 among confirmed baseline/auxiliary sets; tie by mean noisy F1",
                final_seed="42 fixed before looking at test; test evaluates baseline and selected candidate, no test-based tuning",
                final_test_noise="ESC-50 fold5 source IDs disjoint from validation fold4",
                test_features_before_selection=False)


def initialize(study, cfg):
    study = Path(study)
    study.mkdir(parents=True, exist_ok=True)
    p = protocol(cfg)
    path = study / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != p:
            raise ValueError("Frozen protocol changed; use a new study directory")
    else:
        write_json(path, p)
    return study


class CachedDataset(Dataset):
    def __init__(self, directory, scaler, set_id, labels, split=None):
        self.frame = pd.read_csv(Path(directory) / "metadata.csv", keep_default_na=False)
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        archive = Path(directory) / "inputs.npz"
        if file_hash(archive) != manifest["inputs_sha256"]:
            raise ValueError("Corrupted input cache")
        with np.load(archive, allow_pickle=False) as data:
            rows = np.arange(len(self.frame)) if split is None else np.flatnonzero(self.frame.split.to_numpy() == split)
            if not np.array_equal(data["paths"], self.frame.path.to_numpy()):
                raise ValueError("Cache metadata/path alignment changed")
            self.spec = torch.from_numpy(data["spec"][rows].copy())
            self.aux = torch.from_numpy(encode(data["raw"][rows], data["valid"][rows], scaler, set_id))
        self.frame = self.frame.iloc[rows].reset_index(drop=True)
        mapping = {label: i for i, label in enumerate(labels)}
        self.targets = torch.tensor([mapping[v] for v in self.frame.label], dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.spec[index], self.aux[index], self.targets[index], self.frame.path.iloc[index]


def loader(dataset, cfg, device, train=False):
    generator = torch.Generator().manual_seed(cfg.seed)
    sampler = None
    if train:
        counts = dataset.frame.label.value_counts()
        weights = [1.0 / counts[v] for v in dataset.frame.label]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler,
                      num_workers=0, shuffle=False, pin_memory=device.type == "cuda", generator=generator)


@torch.no_grad()
def evaluate(model, batches, device, labels):
    model.eval()
    targets, predictions, paths, probabilities = [], [], [], []
    loss = 0.0
    for spec, aux, y, batch_paths in batches:
        logits = model(spec.to(device), aux.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite evaluation output")
        loss += float(nn.functional.cross_entropy(logits, y.to(device))) * len(y)
        targets.extend(y.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        paths.extend(batch_paths)
        probabilities.extend(logits.softmax(1).cpu().numpy())
    scores = metrics(targets, predictions, len(labels))
    scores.update(loss=loss / len(targets), n=len(targets))
    result = pd.DataFrame(dict(path=paths, true_idx=targets, pred_idx=predictions))
    for j, label in enumerate(labels):
        result[f"p_{j}"] = np.asarray(probabilities)[:, j]
    return scores, result


def lightweight_report(directory, scores, pred, labels):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pred = pred.copy()
    pred["true_label"] = pred.true_idx.map(dict(enumerate(labels)))
    pred["pred_label"] = pred.pred_idx.map(dict(enumerate(labels)))
    write_json(directory / "metrics.json", scores)
    pred.to_csv(directory / "predictions.csv", index=False, encoding="utf-8-sig")
    write_json(directory / "classification_report.json", classification_report(
        pred.true_idx, pred.pred_idx, labels=list(range(11)), target_names=labels,
        output_dict=True, zero_division=0))
    pd.DataFrame(confusion_matrix(pred.true_idx, pred.pred_idx, labels=list(range(11))),
                 index=labels, columns=labels).to_csv(directory / "confusion_counts.csv", encoding="utf-8-sig")


def stress_evaluate(study, run, cfg, model, device, scaler, set_id, split="val"):
    results = []
    for category in NOISE_CATEGORIES:
        for snr in SNRS:
            condition = f"{split}_{category}_{snr}db"
            data = CachedDataset(study / "cache" / condition, scaler, set_id, cfg.labels)
            scores, pred = evaluate(model, loader(data, cfg, device), device, cfg.labels)
            lightweight_report(run / "evaluation" / condition, scores, pred, cfg.labels)
            results.append(dict(condition=condition, category=category, snr_db=snr, **scores))
    name = f"{split}_noise_only"
    data = CachedDataset(study / "cache" / name, scaler, set_id, cfg.labels)
    scores, pred = evaluate(model, loader(data, cfg, device), device, cfg.labels)
    scores["false_keyword_rate"] = float((pred.pred_idx != cfg.labels.index("silence")).mean())
    # Macro-F1 across 11 classes is not informative when every target is silence.
    scores["macro_f1_interpretation"] = "not used for ranking; noise-only has one true class"
    lightweight_report(run / "evaluation" / name, scores, pred, cfg.labels)
    write_json(run / f"{split}_stress_summary.json", dict(conditions=results, noise_only=scores,
               mean_noisy_accuracy=float(np.mean([r["accuracy"] for r in results])),
               mean_noisy_macro_f1=float(np.mean([r["macro_f1"] for r in results]))))


def run_one(study, cfg, set_id, seed=42, fusion="middle", control_extra=0, name=None):
    set_id = str(set_id)
    cfg.seed = int(seed)
    if cfg.use_augmentation:
        raise ValueError("Cached ablation forbids augmentation; implement waveform-first recomputation in a new protocol")
    run_name = name or f"set{set_id}_seed{seed}_{fusion}"
    output = study / "runs" / run_name
    if (output / "summary.json").exists() and (output / "val_stress_summary.json").exists():
        return output
    if output.exists():
        raise ValueError(f"Incomplete run exists; preserve it and use --name to retry: {output}")
    output.mkdir(parents=True)
    cfg.device = "cuda"
    device = choose_device(cfg.device)
    scaler = json.loads((study / "scaler.json").read_text())
    train = CachedDataset(study / "cache" / "original", scaler, set_id, cfg.labels, "train")
    val = CachedDataset(study / "cache" / "original", scaler, set_id, cfg.labels, "val")
    seed_all(seed)
    train_loader, val_loader = loader(train, cfg, device, True), loader(val, cfg, device)
    model = AcousticModel(train.aux.shape[1], cfg.use_pretrained, fusion, control_extra).to(device)
    initial_hash = state_hash(model.backbone.state_dict())
    params = sum(p.numel() for p in model.parameters())
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in
               list((ROOT / "experiments" / "auxiliary").glob("*.py")) + list((ROOT / "src").glob("*.py"))}
    write_json(output / "config.json", dict(base=cfg.to_dict(), set_id=set_id, features=SETS[set_id],
               seed=seed, fusion=fusion, control_extra=control_extra, channels=train.aux.shape[1]))
    write_json(output / "manifest.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
               git=git_info(), source_sha256=sources, protocol_sha256=file_hash(study / "protocol.json"),
               scaler_sha256=file_hash(study / "scaler.json"), original_cache_sha256=file_hash(study / "cache/original/inputs.npz"),
               python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(device), backbone_initial_sha256=initial_hash,
               parameters=params, baseline_head="512 -> 11", test_evaluated=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    history, best_f1, best_epoch = [], -1.0, None
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    log = (output / "epochs.jsonl").open("a")
    try:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total = 0.0
            for spec, aux, y, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(spec.to(device), aux.to(device))
                loss = nn.functional.cross_entropy(prediction, y.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(y)
            scores, _ = evaluate(model, val_loader, device, cfg.labels)
            row = dict(epoch=epoch, learning_rate=optimizer.param_groups[0]["lr"],
                       train_loss=total / len(train), **{f"val_{k}": v for k, v in scores.items()})
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            print(f"{run_name} epoch={epoch} loss={row['train_loss']:.4f} val_acc={scores['accuracy']:.4f} F1={scores['macro_f1']:.4f}", flush=True)
            if scores["macro_f1"] > best_f1:
                best_f1, best_epoch = scores["macro_f1"], epoch
                torch.save(dict(model={k: v.detach().cpu() for k,v in model.state_dict().items()},
                                epoch=epoch, validation=scores), output / "best.pt")
            scheduler.step()
    except Exception as exc:
        write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc),
                   completed_epochs=len(history), test_evaluated=False))
        raise
    finally:
        log.close()
    torch.cuda.synchronize()
    duration = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 1024**2
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    scores, pred = evaluate(model, val_loader, device, cfg.labels)
    save_reports(output / "validation", scores, pred, val.frame, cfg)
    write_json(output / "summary.json", dict(set_id=set_id, seed=seed, fusion=fusion,
               control_extra=control_extra, parameters=params, best_epoch=best_epoch,
               validation=scores, training_seconds=duration, peak_cuda_allocated_mib=peak,
               backbone_initial_sha256=initial_hash, test_evaluated=False))
    stress_evaluate(study, output, cfg, model, device, scaler, set_id)
    del model, optimizer, train, val
    torch.cuda.empty_cache()
    return output


def collect(study):
    rows = []
    for p in sorted((study / "runs").glob("*/summary.json")):
        s = json.loads(p.read_text())
        stress_path = p.parent / "val_stress_summary.json"
        if not stress_path.exists():
            continue
        stress = json.loads(stress_path.read_text())
        rows.append(dict(run=p.parent.name, set_id=str(s["set_id"]), seed=s["seed"],
                         fusion=s["fusion"], control_extra=s["control_extra"],
                         parameters=s["parameters"], best_epoch=s["best_epoch"],
                         val_accuracy=s["validation"]["accuracy"], val_macro_f1=s["validation"]["macro_f1"],
                         mean_noisy_accuracy=stress["mean_noisy_accuracy"],
                         mean_noisy_macro_f1=stress["mean_noisy_macro_f1"],
                         noise_only_false_keyword_rate=stress["noise_only"]["false_keyword_rate"],
                         training_seconds=s["training_seconds"], peak_cuda_mib=s["peak_cuda_allocated_mib"]))
    result = pd.DataFrame(rows)
    if len(result):
        result.to_csv(study / "results.csv", index=False)
    return result


def finalize(study, cfg, workers):
    results = collect(study)
    regular = results[(results.fusion == "middle") & (results.control_extra == 0)]
    confirmed = regular.groupby("set_id").filter(lambda x: set(x.seed) >= {42,43,44})
    if not len(confirmed):
        raise ValueError("Need completed confirmation runs before final selection")
    ranking = confirmed.groupby("set_id")[["val_macro_f1", "mean_noisy_macro_f1"]].mean().sort_values(
        ["val_macro_f1", "mean_noisy_macro_f1"], ascending=False)
    chosen = str(ranking.index[0])
    freeze = dict(selected_set=chosen, seed=42, ranking=ranking.reset_index().to_dict("records"),
                  selected_without_test=True, created_utc=datetime.now(timezone.utc).isoformat())
    reservation = study / "final_selection.json"
    if reservation.exists():
        saved = json.loads(reservation.read_text())
        if saved["selected_set"] != chosen:
            raise ValueError("Final test selection cannot be changed")
    else:
        write_json(reservation, freeze)
    prepare(study, cfg, workers, split="test")
    scaler = json.loads((study / "scaler.json").read_text())
    device = choose_device("cuda")
    for set_id in sorted({"0", chosen}):
        path = study / "runs" / f"set{set_id}_seed42_middle"
        if (path / "test_stress_summary.json").exists():
            continue
        config = json.loads((path / "config.json").read_text())
        seed_all(42)
        model = AcousticModel(config["channels"], pretrained=False).to(device)
        checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        data = CachedDataset(study / "cache/test_original", scaler, set_id, cfg.labels)
        scores, pred = evaluate(model, loader(data, cfg, device), device, cfg.labels)
        save_reports(path / "evaluation/test_original", scores, pred, data.frame, cfg)
        stress_evaluate(study, path, cfg, model, device, scaler, set_id, split="test")
        summary = json.loads((path / "summary.json").read_text())
        summary.update(test_evaluated=True, test=scores)
        write_json(path / "summary.json", summary)
        print(f"FINAL TEST set{set_id}: {scores}", flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["prepare", "run", "screen", "confirm", "collect", "finalize"])
    parser.add_argument("--config", default=str(ROOT / "configs/task2_real.json"))
    parser.add_argument("--study", default=str(ROOT / "runs/auxiliary-v1"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--set", dest="set_id", default="0", choices=list(SETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion", choices=["middle", "late"], default="middle")
    parser.add_argument("--control-extra", type=int, default=0)
    parser.add_argument("--name")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = load_config(args.config)
    study = initialize(args.study, cfg)
    if args.action == "prepare":
        prepare(study, cfg, args.workers)
    elif args.action == "run":
        run_one(study, cfg, args.set_id, args.seed, args.fusion, args.control_extra, args.name)
    elif args.action == "screen":
        for set_id in SETS:
            run_one(study, cfg, set_id)
            collect(study)
    elif args.action == "confirm":
        result = collect(study)
        screen = result[(result.seed == 42) & (result.fusion == "middle") &
                        (result.control_extra == 0) & (result.set_id != "0")]
        selected = screen.sort_values(["val_macro_f1", "mean_noisy_macro_f1"], ascending=False).head(2).set_id.tolist()
        write_json(study / "confirmation_selection.json", dict(sets=["0"] + selected,
                   based_on="screening original validation macro F1, noisy F1 tie break"))
        for set_id in ["0"] + selected:
            for seed in [43, 44]:
                run_one(study, cfg, set_id, seed)
                collect(study)
        # Test fusion and capacity without altering the primary eight-set protocol.
        best = selected[0]
        run_one(study, cfg, best, 42, fusion="late")
        regular = json.loads((study / "runs" / f"set{best}_seed42_middle/summary.json").read_text())
        baseline = json.loads((study / "runs/set0_seed42_middle/summary.json").read_text())
        extra = regular["parameters"] - baseline["parameters"]
        run_one(study, cfg, "0", 42, control_extra=extra, name="capacity_control_seed42")
    elif args.action == "finalize":
        finalize(study, cfg, args.workers)
    collect(study)


if __name__ == "__main__":
    main()
