"""Resumable paired four-speaker CV with the original 512-dimensional features."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from experiments.auxiliary.model import AcousticModel, state_hash
from experiments.auxiliary.study import evaluate, lightweight_report
from src.config import load_config
from src.engine import file_hash, seed_all, write_json
from .data import FoldDataset, Inputs, SPEAKERS, prepare

ROOT = Path(__file__).resolve().parents[2]
BASE = dict(augmentation=[], weight_decay=1e-4, dropout=0.0, freeze_front=False, backbone_lr_ratio=1.0)


def condition(**changes):
    result = dict(BASE, **changes)
    result["augmentation"] = sorted(result["augmentation"])
    return result


def condition_id(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]


class CVModel(AcousticModel):
    def __init__(self, channels, pretrained=True, dropout=0.0, freeze_front=False):
        super().__init__(channels, pretrained)
        self.classifier_dropout = nn.Dropout(dropout)
        self.freeze_front = freeze_front
        if freeze_front:
            for module in (self.backbone.conv1, self.backbone.bn1, self.backbone.layer1):
                for p in module.parameters():
                    p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_front:
            self.backbone.bn1.eval()
            self.backbone.layer1.eval()
        return self

    def forward(self, spec, auxiliary=None):
        return self.backbone.fc(self.classifier_dropout(self.forward_features(spec, auxiliary)))


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_suffix(".partial.pt")
    torch.save(value, temp)
    temp.replace(path)


def batches(data, cfg, train=False, generator=None):
    sampler = None
    if train:
        counts = data.frame.label.value_counts()
        sampler = WeightedRandomSampler([1 / counts[label] for label in data.frame.label],
                                        len(data), replacement=True, generator=generator)
    return DataLoader(data, batch_size=cfg.batch_size, sampler=sampler, shuffle=False,
                      num_workers=0, pin_memory=True,
                      generator=torch.Generator().manual_seed(cfg.seed + 1000))


def run_fold(study, cfg, inputs, settings, set_id, seed, speaker):
    study = Path(study)
    cfg.seed = int(seed)
    name = f"c{condition_id(settings)}_set{set_id}_seed{seed}_{speaker}"
    run = study / "runs" / name
    if (run / "summary.json").exists():
        return json.loads((run / "summary.json").read_text())
    run.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    train = FoldDataset(inputs, speaker, set_id, cfg.labels, True, settings["augmentation"], seed)
    validation = FoldDataset(inputs, speaker, set_id, cfg.labels)
    generator = torch.Generator().manual_seed(seed)
    train_loader = batches(train, cfg, True, generator)
    val_loader = batches(validation, cfg)
    model = CVModel(train.aux.shape[1], cfg.use_pretrained, settings["dropout"],
                    settings["freeze_front"]).cuda()
    initial = state_hash(model.backbone.state_dict())
    back, new = [], []
    for name_p, param in model.named_parameters():
        if param.requires_grad:
            (back if name_p.startswith("backbone.") and not name_p.startswith("backbone.fc.") else new).append(param)
    optimizer = torch.optim.AdamW([
        dict(params=back, lr=cfg.learning_rate * settings["backbone_lr_ratio"]),
        dict(params=new, lr=cfg.learning_rate)], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    configuration = dict(base=cfg.to_dict(), condition=settings, set_id=str(set_id), seed=seed,
                         validation_speaker=speaker, test_evaluated=False)
    config_path = run / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Incomplete run config changed")
    write_json(config_path, configuration)
    sources = {str(p.relative_to(ROOT)): file_hash(p)
               for folder in (ROOT / "experiments/auxiliary_cv", ROOT / "experiments/auxiliary", ROOT / "src")
               for p in folder.glob("*.py")}
    if not (run / "manifest.json").exists():
        write_json(run / "manifest.json", dict(source_sha256=sources,
            protocol_sha256=file_hash(study / "protocol.json"),
            scaler_sha256=file_hash(study / "folds" / speaker / "scaler.json"),
            backbone_initial_sha256=initial, torch=str(torch.__version__),
            gpu=torch.cuda.get_device_name(0), test_evaluated=False))
    history, best, start_epoch, elapsed = [], -1.0, 1, 0.0
    last = run / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, start_epoch = state["history"], state["best_f1"], state["epoch"] + 1
        elapsed = state["elapsed"]
        generator.set_state(state["sampler_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    started = time.monotonic()
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        train.epoch, train.draw = epoch, 0
        loss_sum = 0.0
        for spec, aux, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(spec.cuda(non_blocking=True), aux.cuda(non_blocking=True))
            loss = nn.functional.cross_entropy(logits, targets.cuda(non_blocking=True))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
        scores, _ = evaluate(model, val_loader, torch.device("cuda"), cfg.labels)
        history.append(dict(epoch=epoch, train_loss=loss_sum / len(train),
                            **{f"val_{k}": v for k, v in scores.items()}))
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        if scores["macro_f1"] > best:
            best = scores["macro_f1"]
            atomic_save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                             epoch=epoch, validation=scores), run / "best.pt")
        scheduler.step()
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), epoch=epoch, history=history, best_f1=best,
            elapsed=elapsed + time.monotonic() - started, sampler_rng=generator.get_state(),
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), last)
    best_state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best_state["model"])
    scores, predictions = evaluate(model, val_loader, torch.device("cuda"), cfg.labels)
    lightweight_report(run / "validation", scores, predictions, cfg.labels)
    original_train = FoldDataset(inputs, speaker, set_id, cfg.labels)
    original_train.indices = train.indices
    original_train.frame, original_train.targets = train.frame, train.targets
    train_scores, _ = evaluate(model, batches(original_train, cfg), torch.device("cuda"), cfg.labels)
    summary = dict(run=run.name, condition_id=condition_id(settings), condition=settings,
        set_id=str(set_id), seed=int(seed), validation_speaker=speaker,
        validation=scores, original_train=train_scores, best_epoch=best_state["epoch"],
        training_seconds=elapsed + time.monotonic() - started,
        backbone_initial_sha256=initial, parameters=sum(p.numel() for p in model.parameters()),
        test_evaluated=False)
    write_json(run / "summary.partial.json", summary)
    (run / "summary.partial.json").replace(run / "summary.json")
    last.unlink(missing_ok=True)  # Best checkpoints persist; only active runs need optimizer recovery.
    print(f"DONE {run.name} F1={scores['macro_f1']:.4f} acc={scores['accuracy']:.4f}", flush=True)
    del model, optimizer, train, validation, original_train
    torch.cuda.empty_cache()
    return summary


def run_condition(study, cfg, inputs, settings, sets=("0", "4"), seeds=(42,)):
    return [run_fold(study, cfg, inputs, settings, str(set_id), int(seed), speaker)
            for seed in seeds for set_id in sets for speaker in SPEAKERS]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("prepare", "baseline"))
    parser.add_argument("--study", default="runs/auxiliary-cv-v1")
    parser.add_argument("--config", default="configs/task2_real_context.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = load_config(args.config)
    if args.action == "prepare":
        prepare(args.study, cfg)
    else:
        run_condition(args.study, cfg, Inputs(args.study), condition())


if __name__ == "__main__":
    main()
