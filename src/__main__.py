"""Run from repo root: python -m src {validate,prepare-metadata,train,evaluate,predict}."""
import argparse
from pathlib import Path

from .config import load_config
from .dataset import metadata_draft, read_metadata
from .engine import evaluate_run, predict_file, train


def main():
    parser = argparse.ArgumentParser(description="Task 2 keyword spotting reference baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "train"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--data-root", help="Local or mounted WAV root; overrides config")
        p.add_argument("--metadata", help="Absolute path, or path relative to data root")
        if command == "train":
            p.add_argument("--output", required=True, help="New directory for this run")
            p.add_argument("--device", choices=("auto", "cpu", "cuda"))
            p.add_argument("--epochs", type=int)
    p = sub.add_parser("prepare-metadata")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--run", required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--data-root", help="Relocate identical data without changing saved split")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"))
    p.add_argument("--final-test", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("--run", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    try:
        if args.command in ("train", "validate"):
            cfg = load_config(args.config, args.data_root, args.metadata,
                              getattr(args, "device", None), getattr(args, "epochs", None))
            if args.command == "train":
                train(cfg, args.output)
            else:
                import pandas as pd
                frame = read_metadata(cfg)
                print(pd.crosstab(frame.label, frame.split).reindex(index=cfg.labels))
                print(f"Metadata OK: {len(frame)} clips; dummy={cfg.dummy}")
        elif args.command == "prepare-metadata":
            frame = metadata_draft(Path(args.data_root).expanduser().resolve())
            # Exclusive creation: never overwrite existing annotations.
            with Path(args.output).open("x", encoding="utf-8-sig", newline="") as stream:
                frame.to_csv(stream, index=False)
            print("Draft created. Fill speaker/session IDs, group and split before training.")
        elif args.command == "evaluate":
            evaluate_run(args.run, args.split, args.final_test, args.data_root, args.device)
        else:
            predict_file(args.run, args.wav, args.device)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
