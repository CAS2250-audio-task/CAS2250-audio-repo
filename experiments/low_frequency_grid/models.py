import copy
import torch
from experiments.frequency_local_subset import model_factory
from experiments.frequency_local.models import FrequencyAttention
from experiments.quick_frequency40.models import Frontend,make_models

ARCHITECTURES=('basic','axis','frequency','frequency_attn','axis_attn',
               'axis_frequency','axis_frequency_attn')


def models(seed,architecture):
    assert architecture in ARCHITECTURES
    if architecture=='basic':return model_factory(seed,'basic',False)
    combined=architecture.startswith('axis_frequency')
    variant='low_axis8' if architecture.startswith('axis') and not combined else 'freq_band2'
    result=make_models(seed,variant)
    if combined:
        # Match the existing axis frontend initialization and preserve the
        # canonical two-band FrequencyTimeCNN, classifier and temporal readout.
        torch.manual_seed(seed+2000)
        frontend=Frontend(8,((5,1),(1,3)))
        for model in result.values():model.backbone[0]=copy.deepcopy(frontend)
    if architecture.endswith('_attn'):
        torch.manual_seed(seed+1002)
        attention=FrequencyAttention()
        for model in result.values():model.backbone[1]=copy.deepcopy(attention)
    return result
