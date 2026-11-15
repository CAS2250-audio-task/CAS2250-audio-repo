"""Check ablation invariants, paired-loss gradients and actual GPU step costs."""
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.compact_pooling.objectives import paired_loss, ProjectionHead
from experiments.local_features.models import variant_models
from .models import CNNS, grid_models, FrequencyAttention


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    x = torch.randn(2, 64, 151, device='cuda')
    reference = variant_models(42, 'local')
    rows = []
    for cnn in CNNS:
        for vertical in (False, True):
            models = grid_models(42, cnn, vertical)
            states = [m.backbone.state_dict() for m in models.values()]
            assert all(torch.equal(states[0][k], s[k]) for s in states[1:] for k in states[0])
            for name, model in models.items():
                model.cuda().eval()
                with torch.no_grad():
                    y = model(x)
                    assert y.shape == (2, 11) and torch.isfinite(y).all()
                    if cnn != 'frequency':
                        reference[name].cuda().eval()
                        torch.testing.assert_close(y, reference[name](x), rtol=0, atol=0)
                rows.append(dict(cnn=cnn, vertical=vertical, model=name,
                                 parameters=sum(p.numel() for p in model.parameters())))
            # Same shared paired-loss path as training, including vertical attention.
            model = models['cnn_self_cls'].train()
            projector = ProjectionHead().cuda()
            optimizer = torch.optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=.001)
            batch = torch.randn(32, 64, 151, device='cuda')
            label = torch.arange(32, device='cuda') % 11
            times = []
            for step in range(4):
                torch.cuda.synchronize(); start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                loss, *_ = paired_loss(model, projector, batch, batch + .1*torch.randn_like(batch), label)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if vertical and step > 0:
                    attention = next(m for m in model.modules() if isinstance(m, FrequencyAttention))
                    assert attention.block.attention.q.weight.grad.abs().sum() > 0
                optimizer.step();torch.cuda.synchronize()
                times.append(time.monotonic()-start)
            model.eval()
            with torch.no_grad():
                for _ in range(2): model(batch)
                torch.cuda.synchronize();start=time.monotonic()
                for _ in range(4): model(batch)
                torch.cuda.synchronize();eval_step=(time.monotonic()-start)/4
            # 20 optimizer batches and 9 * 7 validation batches per model/epoch.
            estimate = 3*(20*sum(times[1:])/3 + 63*eval_step)
            for row in rows[-3:]:
                row.update(self_train_step_seconds=sum(times[1:])/3,
                    self_eval_step_seconds=eval_step,
                    estimated_group_epoch_compute_seconds=estimate)
            del models, model, optimizer, projector
            torch.cuda.empty_cache()
            print(json.dumps(rows[-3:]), flush=True)
    root = Path(__file__).resolve().parents[2]
    report = root/'reports/frequency-local-v1'
    report.mkdir(parents=True, exist_ok=True)
    (report/'preflight.json').write_text(json.dumps(dict(
        common_backbones_identical=True, identity_frontends_match_local=True,
        paired_loss_finite=True, vertical_q_gradient_verified=True,
        cost_estimate='Conservative: self head step cost used for all three heads; excludes augmentation and I/O',
        rows=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
