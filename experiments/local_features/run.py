"""Use the established trainer with a model-factory callback and pinned variant metadata."""
import argparse
from pathlib import Path
from experiments.compact_pooling import run as trainer
from .models import variant_models,variant_description,VARIANTS


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    args,remaining=parser.parse_known_args()
    info=variant_description(args.variant)
    root=Path(__file__).resolve().parents[2]
    info['extra_source_files']=[str(p.relative_to(root)) for p in sorted(Path(__file__).parent.glob('*.py'))]
    trainer.main(remaining,model_factory=lambda seed:variant_models(seed,args.variant),experiment_config=info)


if __name__=='__main__':main()
