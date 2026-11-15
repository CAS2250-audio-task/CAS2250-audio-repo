"""Split heads across GPUs without changing canonical classifier/projector initialization."""
import argparse
import copy
from pathlib import Path
import torch
from experiments.compact_pooling import run as trainer
from experiments.compact_pooling.objectives import ProjectionHead
from experiments.frequency_local.models import CNNS, grid_models

NAMES = ('cnn_mean', 'cnn_pooled_query', 'cnn_self_cls')


def model_factory(seed, cnn, vertical):
    models = grid_models(seed, cnn, vertical)
    classifier = models['cnn_mean'].classifier.state_dict()
    for model in models.values():
        model.classifier.load_state_dict(classifier)
    return models


def canonical_projectors():
    # The full trainer creates all three in this order after manual_seed(4242).
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4242)
        return {name: ProjectionHead() for name in NAMES}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cnn', choices=CNNS, required=True)
    parser.add_argument('--vertical', action='store_true')
    parser.add_argument('--models', choices=NAMES, nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    assert len(args.models) == len(set(args.models))
    projectors = canonical_projectors()
    requested = iter(args.models)
    trainer.ProjectionHead = lambda: copy.deepcopy(projectors[next(requested)])
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__)]
    for folder in ('frequency_local', 'local_features'):
        sources += sorted((root/'experiments'/folder).glob('*.py'))
    info = dict(name='frequency-local-split-v1', cnn=args.cnn, vertical=args.vertical,
        local=True, classifier_initialization='canonical cnn_mean classifier shared by all heads',
        projector_initialization='canonical all-head seed4242 sequence, select by original head name',
        frequency_attention_location='after optional 2D frontend, before CNN',
        temporal_support_frames=113 if args.cnn == 'lowcost' else 109,
        local_support_frames=27 if args.cnn == 'lowcost' else 23,
        extra_source_files=[str(p.relative_to(root)) for p in sources])
    trainer.main(remaining + ['--models'] + args.models,
        model_factory=lambda seed: model_factory(seed, args.cnn, args.vertical), experiment_config=info)


if __name__ == '__main__':
    main()
