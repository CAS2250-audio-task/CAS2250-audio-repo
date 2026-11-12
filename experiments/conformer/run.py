"""Frozen and fine-tuned Korean Conformer, preserving reference files and split."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src import engine
from src.config import load_config
from src.dataset import audio_path, read_metadata
from experiments.pretrain.run import export_run, tensor_hash
from .model import ConformerClassifier, REPOSITORY, REVISION, SOURCE_HASHES, load_source

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
MODES = ("frozen", "finetune")


class Clips(Dataset):
    def __init__(self, frame, cfg):
        self.frame = frame.reset_index(drop=True)
        self.wavs = []
        mapping = {label: i for i, label in enumerate(cfg.labels)}
        self.targets = [mapping[label] for label in self.frame.label]
        for relative in self.frame.path:
            wav, sr = sf.read(audio_path(cfg.data_root, relative), dtype="float32")
            if sr != 16000 or wav.shape != (24000,) or not np.isfinite(wav).all():
                raise ValueError(f"Expected unchanged mono 16 kHz / 1.5 s reference clip: {relative}")
            self.wavs.append(torch.from_numpy(wav))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return self.wavs[index], self.targets[index], self.frame.iloc[index].path


def loaders(train, val, seed, batch_size):
    generator = torch.Generator().manual_seed(seed)
    counts = train.frame.label.value_counts()
    weights = [1.0 / counts[label] for label in train.frame.label]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    return (
        DataLoader(train, batch_size=batch_size, sampler=sampler, num_workers=0,
                   generator=generator, pin_memory=True),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0,
                   generator=torch.Generator().manual_seed(seed), pin_memory=True),
        DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
    )


def atomic_save(value, path):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def run_one(mode, seed, state, statistics, cfg, frame, train, val, study, report):
    run = study / "runs" / f"conformer_{mode}_seed{seed}"
    if (run / "comparison_summary.json").exists():
        return json.loads((run / "comparison_summary.json").read_text())
    run.mkdir(parents=True, exist_ok=True)
    cfg.seed = seed
    engine.seed_all(seed)
    train_loader, val_loader, original_train = loaders(train, val, seed, cfg.batch_size)
    model = ConformerClassifier(state, statistics, mode).cuda()
    encoder_state = {k: v for k, v in model.state_dict().items() if not k.startswith("fc.")}
    initialization = dict(
        mode=mode, seed=seed, copied_encoder_tensors=model.copied_tensors,
        encoder_initial_sha256=tensor_hash(encoder_state),
        classifier_initial_sha256=tensor_hash(model.fc.state_dict()),
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        classifier_parameters=sum(p.numel() for p in model.fc.parameters()),
        encoder_dropout_active=(mode == "finetune"),
        source_normalization_updated=False,
    )
    del encoder_state
    if mode == "frozen":
        groups = [{"params": model.fc.parameters(), "lr": cfg.learning_rate}]
    else:
        groups = [
            {"params": list(model.frontend.parameters()) + list(model.transformer.parameters()), "lr": 1e-5},
            {"params": model.fc.parameters(), "lr": cfg.learning_rate},
        ]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    best_f1, best_epoch, history, first_epoch = -1.0, None, [], 1
    latest = run / "last.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, best_f1, best_epoch = checkpoint["history"], checkpoint["best_f1"], checkpoint["best_epoch"]
        first_epoch = checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        train_loader.sampler.generator.set_state(checkpoint["loader_rng"])
        val_loader.generator.set_state(checkpoint["val_loader_rng"])
    else:
        engine.write_json(run / "initialization.json", initialization)
        engine.write_json(run / "reference_config.json", cfg.to_dict())
        frame.to_csv(run / "metadata.csv", index=False, encoding="utf-8-sig")
    print(f"RUN_START {mode} seed={seed} parameters={initialization['total_parameters']} "
          f"trainable={initialization['trainable_parameters']} epoch={first_epoch}", flush=True)
    started = time.monotonic()
    for epoch in range(first_epoch, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
        scores, predictions = engine.evaluate_model(model, val_loader, "cuda", len(cfg.labels))
        row = dict(epoch=epoch, train_loss=total_loss / len(train),
                   encoder_lr=optimizer.param_groups[0]["lr"] if mode == "finetune" else 0.0,
                   head_lr=optimizer.param_groups[-1]["lr"],
                   **{f"val_{k}": v for k, v in scores.items()})
        history.append(row)
        directory = run / "epochs"
        directory.mkdir(exist_ok=True)
        predictions.to_csv(directory / f"epoch_{epoch:02d}.csv", index=False, encoding="utf-8-sig")
        if scores["macro_f1"] > best_f1:
            best_f1, best_epoch = scores["macro_f1"], epoch
            atomic_save(dict(model=model.state_dict(), epoch=epoch, validation=scores,
                             reference_config=cfg.to_dict(), mode=mode), run / "best.pt")
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        print(f"EPOCH {mode} seed={seed} epoch={epoch} train_loss={row['train_loss']:.4f} "
              f"val_acc={scores['accuracy']:.4f} val_f1={scores['macro_f1']:.4f}", flush=True)
        scheduler.step()
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), history=history, epoch=epoch,
                         best_f1=best_f1, best_epoch=best_epoch, torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all(),
                         loader_rng=train_loader.sampler.generator.get_state(),
                         val_loader_rng=val_loader.generator.get_state()), latest)
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    scores, predictions = engine.evaluate_model(model, val_loader, "cuda", len(cfg.labels))
    engine.save_reports(run / "validation", scores, predictions, frame, cfg)
    train_scores, train_predictions = engine.evaluate_model(model, original_train, "cuda", len(cfg.labels))
    train_predictions.to_csv(run / "original_train_predictions.csv", index=False, encoding="utf-8-sig")
    final_encoder = {k: v for k, v in model.state_dict().items() if not k.startswith("fc.")}
    final_hash = tensor_hash(final_encoder)
    saved_init = json.loads((run / "initialization.json").read_text())
    if mode == "frozen" and final_hash != saved_init["encoder_initial_sha256"]:
        raise ValueError("Frozen encoder changed during training")
    result = dict(initialization=f"conformer_{mode}", seed=seed, best_epoch=best_epoch,
                  validation=scores, original_train=train_scores,
                  checkpoint_sha256=engine.file_hash(run / "best.pt"),
                  encoder_final_sha256=final_hash,
                  source_normalization_unchanged=bool(torch.equal(model.feature_mean.cpu(), statistics["glob_mean"])
                       and torch.equal(model.feature_std.cpu(), statistics["glob_std"])),
                  test_evaluated=False, seconds=time.monotonic() - started)
    engine.write_json(run / "comparison_summary.json", result)
    export_run(run, report / "runs" / run.name)
    del model, optimizer, checkpoint
    torch.cuda.empty_cache()
    print("RUN_COMPLETE", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/conformer-research"))
    parser.add_argument("--study", type=Path, default=Path("runs/conformer-ref-v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/conformer-ref-v1"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise ValueError("This protocol requires the allocated CUDA GPU")
    state, statistics = load_source(args.source)
    cfg = load_config(ROOT / "configs/task2_real.json")
    assert cfg.epochs == 10 and cfg.batch_size == 32 and not cfg.use_augmentation
    frame = read_metadata(cfg)
    assert frame.groupby("split").size().to_dict() == {"test": 203, "train": 609, "val": 203}
    args.study.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    hashes = {p: engine.file_hash(audio_path(cfg.data_root, p)) for p in frame.path}
    protocol = dict(
        base_ref_commit="6d208bc", reference_config=cfg.to_dict(),
        seeds=list(SEEDS), conditions=list(MODES), epochs=10,
        source=dict(repository=REPOSITORY, revision=REVISION, files=SOURCE_HASHES,
                    corpus="KsponSpeech", source_hours=965.2),
        data=dict(train_speakers=["spk02", "spk03", "spk04"], validation_speaker="spk05",
                  train_files=609, validation_files=203, test_files=203,
                  metadata_sha256=engine.file_hash(Path(cfg.data_root) / cfg.metadata_csv),
                  audio_sha256=hashes, input_samples=24000, test_evaluated=False),
        model=dict(frontend="source 3-block CNN", encoder="12-layer Korean Conformer, hidden 256",
                   feature="native SpeechBrain Fbank, 80 bins, FFT400, window25ms, hop10ms, f_min0",
                   normalization="fixed pretrained global mean/std; no target-data updates",
                   pooling="mean across all frames of unchanged 1.5-second clip",
                   classifier=[256,11], extra_audio_padding_samples=0,
                   source_dropout=0.1),
        optimizer=dict(name="AdamW", head_lr=5e-4, finetune_encoder_lr=1e-5,
                       weight_decay=1e-4, scheduler="cosine over 10 epochs",
                       sampler="inverse-class weighted replacement, 609 draws/epoch"),
        environment={name:version(name) for name in ("speechbrain", "torch", "torchaudio", "numpy",
                      "huggingface-hub", "hyperpyyaml", "sentencepiece", "ruamel.yaml", "hf-xet")},
        source_sha256={str(p.relative_to(ROOT)):engine.file_hash(p) for p in
                       [*sorted((ROOT/"src").glob("*.py")), *sorted(Path(__file__).parent.glob("*.py"))]},
        interpretation="System comparison: encoder architecture, native features and fine-tuning LR differ from ResNet."
    )
    path = args.study / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol changed; use a new study directory")
    engine.write_json(path, protocol)
    engine.write_json(args.report / "protocol.json", protocol)
    train = Clips(frame[frame.split == "train"], cfg)
    val = Clips(frame[frame.split == "val"], cfg)
    engine.seed_all(42)
    probe = ConformerClassifier(state, statistics).cuda().eval()
    with torch.no_grad():
        wav = torch.stack([train[0][0], train[1][0]]).cuda()
        features = probe.features(wav)
        encoded = probe.encode(wav)
        logits = probe(wav)
        shorter = probe(wav[:, :16000])
        assert features.shape == (2,151,80) and encoded.shape == (2,38,256)
        assert logits.shape == (2,11) and shorter.shape == (2,11)
        assert torch.isfinite(logits).all() and torch.isfinite(shorter).all()
    engine.write_json(args.report / "input_probe.json", dict(
        input_shape=list(wav.shape), feature_shape=list(features.shape), encoded_shape=list(encoded.shape),
        output_shape=list(logits.shape), one_second_input_accepted=True,
        no_extra_audio_padding=True, source_normalizer_count=int(statistics["count"]),
        encoder_tensors_loaded=probe.copied_tensors,
        total_parameters=sum(p.numel() for p in probe.parameters())))
    del probe, wav, features, encoded, logits, shorter
    torch.cuda.empty_cache()
    results = []
    for seed in args.seeds:
        for mode in args.modes:
            results.append(run_one(mode, seed, state, statistics, cfg, frame, train, val, args.study, args.report))
        if len(args.modes) == 2:
            audits = [json.loads((args.study/"runs"/f"conformer_{mode}_seed{seed}"/"initialization.json").read_text())
                      for mode in args.modes]
            assert audits[0]["classifier_initial_sha256"] == audits[1]["classifier_initial_sha256"]
            assert audits[0]["encoder_initial_sha256"] == audits[1]["encoder_initial_sha256"]
    engine.write_json(args.study / "complete.json", dict(
        runs=len(results), epochs=sum(len(pd.read_csv(args.study/"runs"/f"{r['initialization']}_seed{r['seed']}"/"history.csv"))
                                     for r in results), test_evaluated=False,
        complete=(set(args.seeds)==set(SEEDS) and set(args.modes)==set(MODES))))


if __name__ == "__main__":
    main()
