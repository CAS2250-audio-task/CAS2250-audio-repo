import argparse,copy
from pathlib import Path
from . import trainer
from .models import VARIANTS,make_models
from experiments.frequency_local_subset import canonical_projectors


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    args,remaining=parser.parse_known_args()
    projector=canonical_projectors()['cnn_pooled_query']
    trainer.ProjectionHead=lambda:copy.deepcopy(projector)
    root=Path(__file__).resolve().parents[2]
    paths=[root/'experiments/frequency_local_subset.py']
    for folder in ['quick_frequency40','frequency_local','local_features','compact_pooling']:
        paths.extend((root/'experiments'/folder).glob('*.py'))
    info=dict(name='quick-frequency40',variant=args.variant,vertical_attention=False,
        pilot='300 epochs at first300 of 1000epoch cosine schedule; compare pilots only',
        initialization='Canonical Pooled classifier/projector; identity or equal-output initialization for added routes',
        extra_source_files=[str(p.relative_to(root)) for p in sorted(set(paths))])
    trainer.main(remaining,model_factory=lambda seed:make_models(seed,args.variant),experiment_config=info)


if __name__=='__main__':main()
