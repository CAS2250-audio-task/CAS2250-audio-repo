"""Training, checkpoint evaluation, prediction and reproducible run artifacts."""
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn.functional as F

from .config import Config
from .dataset import AudioDataset, audio_path, make_loader, read_metadata, validate_metadata
from .model import build_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Check the venv and NVIDIA driver.")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info():
    root = Path(__file__).resolve().parent.parent
    def run(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "status": run("status", "--short")}


def metrics(targets, preds, num_classes):
    indices = list(range(num_classes))
    return {"accuracy": float(accuracy_score(targets, preds)),
            "macro_f1": float(f1_score(targets, preds, labels=indices, average="macro", zero_division=0))}


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes):
    model.eval()
    targets, predictions, paths = [], [], []
    total_loss = 0.0
    for x, y, batch_paths in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(F.cross_entropy(logits, y)) * len(y)
        targets.extend(y.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        paths.extend(batch_paths)
    result = metrics(targets, predictions, num_classes)
    result["loss"] = total_loss / len(targets)
    return result, pd.DataFrame({"path": paths, "true_idx": targets, "pred_idx": predictions})


def save_reports(directory, scores, predictions, frame, cfg):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "metrics.json", scores)
    labels = cfg.labels
    indices = list(range(len(labels)))
    predictions = predictions.copy()
    predictions["true_label"] = predictions.true_idx.map(dict(enumerate(labels)))
    predictions["pred_label"] = predictions.pred_idx.map(dict(enumerate(labels)))
    predictions.to_csv(directory / "predictions.csv", index=False, encoding="utf-8-sig")
    errors = predictions[predictions.true_idx != predictions.pred_idx].copy()
    errors["error_reason"] = ""
    errors.to_csv(directory / "error_analysis.csv", index=False, encoding="utf-8-sig")
    report = classification_report(predictions.true_idx, predictions.pred_idx,
                                   labels=indices, target_names=labels, zero_division=0, output_dict=True)
    write_json(directory / "classification_report.json", report)
    matrix = confusion_matrix(predictions.true_idx, predictions.pred_idx, labels=indices)
    denominator = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, denominator, out=np.zeros_like(matrix, dtype=float), where=denominator != 0)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(directory / "confusion_counts.csv", encoding="utf-8-sig")
    pd.DataFrame(normalized, index=labels, columns=labels).to_csv(directory / "confusion_normalized.csv", encoding="utf-8-sig")
    # Numeric class IDs keep plots portable on servers without Korean fonts.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=indices, yticks=indices, xlabel="Predicted class ID", ylabel="True class ID",
           title="Normalized confusion matrix (label order: config.json)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png")
    plt.close(fig)
    merged = predictions.merge(frame, on="path", validate="one_to_one")
    grouped = []
    for column in ("speaker_id", "room_session_id", "room", "distance_cm", "noise_level", "noise_type", "device", "style", "group"):
        if column not in merged:
            continue
        for value, subset in merged[merged[column].astype(str).str.strip() != ""].groupby(column):
            present = sorted(subset.true_idx.unique().tolist())
            grouped.append({"dimension": column, "value": str(value), "n": len(subset),
                            "accuracy": float(accuracy_score(subset.true_idx, subset.pred_idx)),
                            # Include every class present in this slice; document different denominators.
                            "macro_f1": float(f1_score(subset.true_idx, subset.pred_idx, labels=present,
                                                       average="macro", zero_division=0)),
                            "class_ids": present})
    write_json(directory / "group_metrics.json", grouped)


def train(cfg, output):
    frame = read_metadata(cfg)
    device = choose_device(cfg.device)
    seed_all(cfg.seed)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg.to_dict())
    frame.to_csv(output / "metadata.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(frame.label, frame.split).reindex(index=cfg.labels).to_csv(output / "split_counts.csv", encoding="utf-8-sig")
    # Fingerprint files for identity/version checking; this does not evaluate test audio.
    hashes = {p: file_hash(audio_path(cfg.data_root, p)) for p in frame.path}
    write_json(output / "manifest.json", {
        "task": "task2_keyword_spotting", "dummy": cfg.dummy,
        "dataset_version": cfg.dataset_version, "split_version": cfg.split_version,
        "metadata_sha256": file_hash(output / "metadata.csv"), "audio_sha256": hashes,
        "git": git_info(), "command": sys.argv, "python": platform.python_version(),
        "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": cfg.seed})
    loaders = {split: make_loader(frame, cfg, split, device) for split in ("train", "val")}
    model = build_model(len(cfg.labels), cfg.use_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    history, best_f1, best_epoch = [], -1.0, None
    print(f"device={device}; dummy={cfg.dummy}; output={output}", flush=True)
    if cfg.dummy:
        print("Synthetic dummy data: scores are only a pipeline check, not project results.", flush=True)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for x, y, _ in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is nonfinite.")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
        val, _ = evaluate_model(model, loaders["val"], device, len(cfg.labels))
        history.append({"epoch": epoch, "train_loss": total / len(loaders["train"].dataset),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.4f} "
              f"val_acc={val['accuracy']:.3f} val_macro_f1={val['macro_f1']:.3f}", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "labels": cfg.labels, "config": cfg.to_dict(), "epoch": epoch,
                        "validation": val}, output / "best.pt")
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        scheduler.step()
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    scores, predictions = evaluate_model(model, loaders["val"], device, len(cfg.labels))
    save_reports(output / "validation", scores, predictions, frame, cfg)
    write_json(output / "summary.json", {"best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                                        "dummy": cfg.dummy, "test_evaluated": False})
    return output


def load_run(run, data_root=None, device=None):
    run = Path(run).expanduser().resolve()
    cfg = Config(**json.loads((run / "config.json").read_text(encoding="utf-8")))
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["labels"] != cfg.labels or checkpoint["config"] != cfg.to_dict():
        raise ValueError("Run config does not match its checkpoint.")
    if data_root is not None:
        cfg.data_root = str(Path(data_root).expanduser().resolve())
    if device is not None:
        cfg.device = device
    cfg.validate()
    selected = choose_device(cfg.device)
    seed_all(cfg.seed)
    # Restoring a checkpoint must never download pretrained weights again.
    model = build_model(len(cfg.labels), use_pretrained=False).to(selected)
    model.load_state_dict(checkpoint["model"])
    return run, cfg, model, selected


def evaluate_run(run, split="val", final_test=False, data_root=None, device=None):
    run, cfg, model, selected = load_run(run, data_root, device)
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test.")
    if split == "test" and not cfg.dummy and not final_test:
        raise ValueError("Use --final-test only after data, model and settings are frozen.")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(run / "metadata.csv") != manifest["metadata_sha256"]:
        raise ValueError("Saved split/metadata has changed. Start a new versioned run.")
    frame = validate_metadata(pd.read_csv(run / "metadata.csv", dtype=str, keep_default_na=False), cfg)
    for path, expected in manifest["audio_sha256"].items():
        if file_hash(audio_path(cfg.data_root, path)) != expected:
            raise ValueError(f"Dataset has changed since training: {path}. Start a new versioned run.")
    directory = run / "evaluation" / split
    if split == "test":
        # Atomic reservation prevents two final-test commands evaluating the same run concurrently.
        directory.mkdir(parents=True, exist_ok=False)
    scores, predictions = evaluate_model(model, make_loader(frame, cfg, split, selected), selected, len(cfg.labels))
    save_reports(directory, scores, predictions, frame, cfg)
    write_json(directory / "evaluation.json", {"split": split, "dummy": cfg.dummy,
               "final_test": final_test, "checkpoint_sha256": file_hash(run / "best.pt"), "device": str(selected)})
    if split == "test":
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        summary["test_evaluated"] = True
        summary["test_report"] = "evaluation/test"
        write_json(run / "summary.json", summary)
    print(json.dumps({"dummy": cfg.dummy, "split": split, **scores}, ensure_ascii=False), flush=True)
    return scores


def predict_file(run, wav, device=None):
    _, cfg, model, selected = load_run(run, device=device)
    dataset = AudioDataset(pd.DataFrame(), cfg, train=False)
    x = dataset.transform_file(Path(wav).expanduser().resolve()).unsqueeze(0).to(selected)
    model.eval()
    with torch.no_grad():
        probabilities = model(x).softmax(1)[0].cpu().tolist()
    index = int(np.argmax(probabilities))
    result = {"label": cfg.labels[index], "dummy": cfg.dummy,
              "probabilities": dict(zip(cfg.labels, probabilities))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
