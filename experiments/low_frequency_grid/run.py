import argparse,copy
from pathlib import Path
from . import trainer
from experiments.frequency_local_subset import canonical_projectors
from .models import models,ARCHITECTURES
from .frontend import Frontend,ALPHAS,configuration


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--architecture',choices=ARCHITECTURES,required=True)
    parser.add_argument('--frontend',choices=list(ALPHAS),required=True)
    args,remaining=parser.parse_known_args()
    trainer.Frontend=lambda:Frontend(args.frontend)
    projector=canonical_projectors()['cnn_pooled_query']
    trainer.ProjectionHead=lambda:copy.deepcopy(projector)
    root=Path(__file__).resolve().parents[2]
    paths=[root/'experiments/frequency_local_subset.py']
    for folder in ['low_frequency_grid','quick_frequency40','frequency_local','local_features','compact_pooling']:
        paths.extend((root/'experiments'/folder).glob('*.py'))
    info=dict(name='low-frequency-grid',architecture=args.architecture,frontend=configuration(args.frontend),
        pilot='Staged training under a fixed 1000epoch cosine protocol; stop budget is runtime-only; Pooled fixed',
        temporal_readout='Pooled',vertical_attention=args.architecture.endswith('_attn'),
        frequency_cnn='2-band frequency pooling in both final and local path' if args.architecture.startswith('frequency') else None,
        source_files='Full source hashes include baseline modules and this isolated wrapper',
        extra_source_files=[str(p.relative_to(root)) for p in sorted(set(paths))])
    trainer.main(remaining+['--models','cnn_pooled_query'],
        model_factory=lambda seed:models(seed,args.architecture),experiment_config=info)


if __name__=='__main__':main()
