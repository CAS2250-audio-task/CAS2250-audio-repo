"""Portable wrapper; all augmentation, loss and evaluation use the shared trainer."""
import argparse
from pathlib import Path
from experiments.compact_pooling import run as trainer
from .models import CNNS, grid_models


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cnn', choices=CNNS, required=True)
    parser.add_argument('--vertical', action='store_true')
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parents[2]
    sources = sorted(Path(__file__).parent.glob('*.py'))
    sources += sorted((root/'experiments/local_features').glob('*.py'))
    info = dict(name='frequency-local-v1', cnn=args.cnn, vertical=args.vertical,
        local=True, frequency_attention_location='after optional 2D frontend, before CNN',
        frequency_attention='64 frequency tokens per frame, d16, heads2, FF32, residual, SDPA' if args.vertical else None,
        temporal_support_frames=113 if args.cnn == 'lowcost' else 109,
        local_support_frames=27 if args.cnn == 'lowcost' else 23,
        frequency_cnn_local='block2: frequency mean, 48->32 residual' if args.cnn == 'frequency' else 'block2:64->32 residual',
        extra_source_files=[str(p.relative_to(root)) for p in sources])
    trainer.main(remaining, model_factory=lambda seed: grid_models(seed, args.cnn, args.vertical), experiment_config=info)


if __name__ == '__main__':
    main()
