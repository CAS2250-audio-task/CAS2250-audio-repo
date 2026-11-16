import copy
import torch
from experiments.frequency_local_subset import model_factory
from experiments.frequency_local.models import FrequencyAttention
from experiments.quick_frequency40.models import make_models

ARCHITECTURES=('basic','axis','frequency','frequency_attn','axis_attn')


def models(seed,architecture):
    assert architecture in ARCHITECTURES
    if architecture=='basic':return model_factory(seed,'basic',False)
    variant='low_axis8' if architecture.startswith('axis') else 'freq_band2'
    result=make_models(seed,variant)
    if architecture.endswith('_attn'):
        torch.manual_seed(seed+1002)
        attention=FrequencyAttention()
        for model in result.values():model.backbone[1]=copy.deepcopy(attention)
    return result
