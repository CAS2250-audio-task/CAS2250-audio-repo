"""Rebuild the selected historical weak-formant bank from development originals."""
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from .run import DATA, OUT, formant_bank

def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(DATA/'metadata.csv')
    frame=frame[frame.speaker_id.isin(['spk02','spk03','spk04','spk05'])].reset_index(drop=True)
    assert len(frame)==812
    waves=np.stack([sf.read(DATA/p,dtype='float32')[0] for p in frame.path])
    bank=formant_bank(frame,waves)
    print('FORMANTS_COMPLETE',bank.shape,flush=True)

if __name__=='__main__':main()
