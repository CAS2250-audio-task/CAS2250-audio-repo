"""Native Conformer with optional aligned auxiliary residual; original head retained."""
import torch
from torch import nn
from experiments.conformer_aug.model import AugmentedConformer
from .features import SETS


class AcousticConformer(AugmentedConformer):
    def __init__(self,state,statistics,speaker,dropout=.1,set_id="0",cache=None,scaler=None,augmented=False):
        super().__init__(state,statistics,speaker,mask=augmented)
        self.set_id=str(set_id);self.cache=cache;self.scaler=scaler
        self.dropout_changes=[]
        for name,module in self.transformer.named_modules():
            if isinstance(module,nn.Dropout):
                self.dropout_changes.append(name);module.p=dropout
            elif isinstance(module,nn.MultiheadAttention):module.dropout=dropout
        assert self.dropout_changes
        channels=2*len(SETS[self.set_id])
        self.auxiliary=None
        if channels:
            self.auxiliary=nn.Sequential(nn.Conv1d(channels,32,5,stride=2,padding=2),nn.GELU(),
                nn.Conv1d(32,64,5,stride=2,padding=2),nn.GELU(),nn.Conv1d(64,256,1))
            nn.init.zeros_(self.auxiliary[-1].weight);nn.init.zeros_(self.auxiliary[-1].bias)

    def forward(self,wave):
        auxiliary=None
        if self.auxiliary is not None:
            assert self.cache is not None and self.scaler is not None
            a=self.cache.encoded(wave.detach().cpu().numpy(),self.scaler,self.set_id)
            auxiliary=torch.from_numpy(a).to(wave.device)
        before=len(self.mask_trace)
        encoded=self.encode(wave)
        if auxiliary is not None:
            # Match time masks in both input streams on the common151-frame grid.
            for row in self.mask_trace[before:]:
                auxiliary[row["item"],:,row["time_start"]:row["time_start"]+row["time_width"]]=0
            residual=self.auxiliary(auxiliary).transpose(1,2)
            assert residual.shape==encoded.shape,(residual.shape,encoded.shape)
            encoded=encoded+residual
        return self.fc(encoded.mean(dim=1))
