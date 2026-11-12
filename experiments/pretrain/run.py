"""Paired ImageNet/SC09 initialization using the unchanged reference trainer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import pandas as pd
import torch

from src import engine
from src.config import load_config
from src.dataset import make_loader, read_metadata
from src.model import build_model as reference_model
from .weights import EXPECTED_SHA256, SOURCE_URL, download_checkpoint, encoder_weights, transfer

ROOT = Path(__file__).resolve().parents[2]


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def export_run(run, destination):
    for path in run.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".csv", ".png"):
            target = destination / path.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def run_one(kind, seed, encoded, source_info, study, report):
    cfg = load_config(ROOT / "configs/task2_real.json")
    cfg.seed, cfg.device = seed, "cuda"
    assert cfg.dataset_version == "task2-1015-fixed-1p5-v1"
    assert cfg.epochs == 10 and cfg.use_pretrained and not cfg.use_augmentation
    run = study / "runs" / f"{kind}_seed{seed}"
    if (run / "comparison_summary.json").exists():
        return json.loads((run / "comparison_summary.json").read_text())
    if run.exists():
        # Keep interrupted artifacts; the reference trainer creates a fresh output.
        run.rename(run.with_name(run.name + f"_interrupted_{time.time_ns()}"))
    saved_build, saved_evaluate = engine.build_model, engine.evaluate_model
    evaluation_count = 0

    def factory(num_classes, use_pretrained):
        model = reference_model(num_classes, use_pretrained=(kind == "imagenet"))
        audit = dict(initialization=kind, source=source_info if kind == "speech_sc09" else
                     dict(weights="torchvision.ResNet18_Weights.IMAGENET1K_V1"))
        if kind == "speech_sc09":
            audit.update(transfer(model, encoded))
        audit.update(classifier_initial_sha256=tensor_hash(model.fc.state_dict()),
                     backbone_initial_sha256=tensor_hash({k: v for k, v in model.state_dict().items()
                                                          if not k.startswith("fc.")}),
                     parameters=sum(p.numel() for p in model.parameters()))
        engine.write_json(run / "initialization.json", audit)
        return model

    def evaluate(model, loader, device, num_classes):
        nonlocal evaluation_count
        scores, predictions = saved_evaluate(model, loader, device, num_classes)
        if len(predictions) == 203:
            evaluation_count += 1
            if evaluation_count <= cfg.epochs:
                directory = run / "epochs"
                directory.mkdir(exist_ok=True)
                result = predictions.copy()
                result["true_label"] = result.true_idx.map(dict(enumerate(cfg.labels)))
                result["pred_label"] = result.pred_idx.map(dict(enumerate(cfg.labels)))
                result.to_csv(directory / f"epoch_{evaluation_count:02d}.csv", index=False, encoding="utf-8-sig")
                engine.write_json(directory / f"epoch_{evaluation_count:02d}_metrics.json", scores)
        return scores, predictions

    engine.build_model, engine.evaluate_model = factory, evaluate
    try:
        engine.train(cfg, run)
    finally:
        engine.build_model, engine.evaluate_model = saved_build, saved_evaluate
    _, restored_cfg, model, device = engine.load_run(run, device="cuda")
    metadata = read_metadata(restored_cfg)
    train_loader = make_loader(metadata, restored_cfg, "train", device)
    # Evaluate each original training file once, without weighted resampling.
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=cfg.batch_size,
                                               shuffle=False, num_workers=0)
    train_scores, _ = saved_evaluate(model, train_loader, device, len(cfg.labels))
    validation = json.loads((run / "validation/metrics.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    result = dict(initialization=kind, seed=seed, best_epoch=summary["best_epoch"],
        validation=validation, original_train=train_scores,
        checkpoint_sha256=engine.file_hash(run / "best.pt"), test_evaluated=False)
    engine.write_json(run / "comparison_summary.json", result)
    export_run(run, report / "runs" / run.name)
    del model
    torch.cuda.empty_cache()
    print("PAIRED_RUN_COMPLETE", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/pretrain-research/clar-supervised-sc09-epoch1000.ckpt"))
    parser.add_argument("--study", type=Path, default=Path("runs/speech-pretrain-ref-v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/speech-pretrain-ref-v1"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.study.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    checkpoint = download_checkpoint(args.checkpoint)
    encoded, source_info = encoder_weights(checkpoint)
    source_info.update(checkpoint_sha256=EXPECTED_SHA256, source_url=SOURCE_URL)
    cfg = load_config(ROOT / "configs/task2_real.json")
    frame = read_metadata(cfg)
    assert frame.groupby("split").size().to_dict() == {"test": 203, "train": 609, "val": 203}
    assert frame[frame.split == "val"].group.eq("speaker:spk05").all()
    sources = {str(p.relative_to(ROOT)): engine.file_hash(p) for p in
               [*sorted((ROOT / "src").glob("*.py")), *sorted(Path(__file__).parent.glob("*.py"))]}
    protocol = dict(base_ref_commit="6d208bc", source_initialization=source_info,
        config=cfg.to_dict(), seeds=[42, 43, 44], conditions=["imagenet", "speech_sc09"],
        dataset_metadata_sha256=engine.file_hash(Path(cfg.data_root) / cfg.metadata_csv),
        source_sha256=sources, same_reference_data_split_model_trainer=True,
        source_input_channels=["STFT_magnitude", "STFT_phase", "log_mel"],
        target_input="unchanged reference one-channel log-mel", first_conv_transfer="select source channel 2",
        test_excluded=True, no_auxiliary_features=True)
    path = args.study / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol changed; use a new study directory")
    engine.write_json(path, protocol)
    engine.write_json(args.report / "protocol.json", protocol)
    results = []
    for seed in (42, 43, 44):
        for kind in ("imagenet", "speech_sc09"):
            results.append(run_one(kind, seed, encoded, source_info, args.study, args.report))
        audits = [json.loads((args.study / "runs" / f"{kind}_seed{seed}" / "initialization.json").read_text())
                  for kind in ("imagenet", "speech_sc09")]
        assert audits[0]["classifier_initial_sha256"] == audits[1]["classifier_initial_sha256"]
    engine.write_json(args.study / "complete.json", dict(complete=True, runs=6, epochs=60,
        same_classifier_initialization_per_seed=True, test_evaluated=False))
    print("STUDY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
