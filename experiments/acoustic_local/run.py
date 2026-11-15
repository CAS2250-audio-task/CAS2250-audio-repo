"""Set3 auxiliary-input experiment; unchanged established training mathematics."""
import argparse
import atexit
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from experiments.compact_pooling import run as trainer
from .features import Extractor,fit_scaler,CONFIG
from .models import acoustic_models


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--aux-workers',type=int,default=4)
    args,remaining=parser.parse_known_args()
    # Single explicit fold avoids accidentally sharing one fold's scaler with another.
    control=argparse.ArgumentParser(add_help=False);control.add_argument('--folds',nargs='+',default=['spk03']);control.add_argument('--output',type=Path,required=True)
    selected,_=control.parse_known_args(remaining);assert selected.folds==['spk03']
    root=Path(__file__).resolve().parents[2];report=root/'reports'/selected.output.name;report.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(trainer.DATA/'metadata.csv');frame=frame[frame.speaker_id.isin(['spk02','spk04','spk05'])]
    assert len(frame)==609 and not frame.path.str.contains('spk01').any()
    extractor=Extractor(args.aux_workers);atexit.register(extractor.close)
    started=time.monotonic()
    waves=np.stack([sf.read(trainer.DATA/p,dtype='float32')[0] for p in frame.path])
    print('AUX preparing original training-only scaler',flush=True)
    encoded=extractor(waves);scaler=fit_scaler(encoded)
    trainer.write(report/'auxiliary_scaler.json',scaler)
    original=trainer.features
    @torch.no_grad()
    def features(frontend,waves,mean=None,std=None):
        mel=original(frontend,waves,mean,std)
        if mean is None:return mel
        aux=torch.from_numpy(extractor(waves)).to(mel.device)
        assert aux.shape==(len(waves),12,151) and torch.isfinite(aux).all()
        return torch.cat([mel,aux],dim=1)
    trainer.features=features
    paths=sorted(Path(__file__).parent.glob('*.py'))+sorted((root/'experiments/local_features').glob('*.py'))
    info=dict(name='local-acoustic-set3',variant='local',auxiliary='F1/F2/F3 and B1/B2/B3 plus six availability channels',
        auxiliary_config=CONFIG,auxiliary_scaler=scaler,auxiliary_fusion='12->16->32->32 temporal Conv1d k5/k5/k1,GroupNorm,GELU; zero-init final projection; add to32D features before readout',
        inputs='Each auxiliary extracted from actual input waveform, including mixed noise; both reference and noisy views; no clean-side auxiliary for noisy input',
        scaler_seconds=time.monotonic()-started,implementation='set3-only selective extraction matches legacy Burg/support settings; does not compute discarded pitch',
        extra_source_files=[str(p.relative_to(root)) for p in paths])
    # Runtime duration must not enter the fixed resume protocol.
    info.pop('scaler_seconds')
    print('AUX scaler ready',time.monotonic()-started,flush=True)
    trainer.main(remaining,model_factory=lambda seed:acoustic_models(seed,scaler),experiment_config=info)
    trainer.write(report/'auxiliary_cache_stats.json',dict(hits=extractor.hits,misses=extractor.misses,max_entries=8192))


if __name__=='__main__':main()
