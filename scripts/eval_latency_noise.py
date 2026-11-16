import json, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import accuracy_score, f1_score
from src.config import Config
from src.model import build_model
from src.dataset import read_metadata, audio_path, AudioDataset

runs = ["seojoon-baseline-v1", "seojoon-exp4_combo", "seojoon-exp5_mobilenet", "seojoon-exp5_mobilenet_tuned"]
rows = []

for r in runs:
    run_dir = Path(f"runs/{r}")
    raw_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = Config(**{k: v for k, v in raw_cfg.items() if k in Config.__dataclass_fields__})
    cfg.validate()
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    in_ch = 3 if cfg.use_delta else 1
    model = build_model(len(cfg.labels), use_pretrained=False, in_channels=in_ch, backbone=cfg.backbone)
    model.load_state_dict(ckpt["model"])
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    dummy = torch.randn(1, in_ch, cfg.n_mels, 151)
    with torch.no_grad():
        for _ in range(15):
            model(dummy)
        t_cpu = []
        for _ in range(100):
            t0 = time.perf_counter()
            model(dummy)
            t_cpu.append((time.perf_counter() - t0) * 1000)

    model_gpu = model.cuda()
    dummy_gpu = dummy.cuda()
    with torch.no_grad():
        for _ in range(20):
            model_gpu(dummy_gpu)
        torch.cuda.synchronize()
        t_gpu = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model_gpu(dummy_gpu)
            torch.cuda.synchronize()
            t_gpu.append((time.perf_counter() - t0) * 1000)

    frame = read_metadata(cfg)
    val_df = frame[frame.split == "val"].reset_index(drop=True)
    ds = AudioDataset(val_df, cfg, train=False)
    clean_m = json.loads((run_dir / "validation/metrics.json").read_text())

    rng = np.random.default_rng(42)
    y_true, y_pred = [], []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            p = audio_path(cfg.data_root, row.path)
            audio, sr = sf.read(p, dtype="float32", always_2d=True)
            wav = torch.from_numpy(audio.T).mean(0, keepdim=True)
            if sr != cfg.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)
            samples = int(cfg.sample_rate * cfg.seconds)
            if wav.shape[-1] < samples:
                wav = F.pad(wav, (0, samples - wav.shape[-1]))
            elif wav.shape[-1] > samples:
                start = (wav.shape[-1] - samples) // 2
                wav = wav[..., start:start + samples]
            rms = wav.pow(2).mean().sqrt().clamp(min=1e-6)
            noise = torch.from_numpy(rng.standard_normal(wav.shape).astype("float32"))
            noise = noise / noise.pow(2).mean().sqrt().clamp(min=1e-6) * rms
            wav_noisy = (wav + noise).clamp(-1, 1)
            if cfg.pre_emphasis > 0:
                wav_noisy = torch.cat([wav_noisy[..., :1], wav_noisy[..., 1:] - cfg.pre_emphasis * wav_noisy[..., :-1]], dim=-1)
            spec = ((ds.to_db(ds.mel(wav_noisy)) + 80.0) / 80.0).clamp(0, 1)
            norm = (spec - 0.5) / 0.25
            if cfg.use_delta:
                d1 = torchaudio.functional.compute_deltas(norm)
                d2 = torchaudio.functional.compute_deltas(d1)
                norm = torch.cat([norm, d1 * 2.0, d2 * 4.0], dim=0)
            logits = model_gpu(norm.unsqueeze(0).cuda())
            y_true.append(ds.label_to_idx[row.label])
            y_pred.append(int(logits.argmax(dim=1).item()))

    noisy_acc = accuracy_score(y_true, y_pred)
    noisy_f1 = f1_score(y_true, y_pred, labels=list(range(len(cfg.labels))), average="macro", zero_division=0)
    rows.append((r, params / 1e6, clean_m["accuracy"]*100, clean_m["macro_f1"]*100, noisy_acc*100, noisy_f1*100, float(np.median(t_cpu)), float(np.median(t_gpu))))

print(f"\n{'Model':<28} | {'Params':>7} | {'CleanAcc':>8} | {'CleanF1':>8} | {'0dB Acc':>8} | {'0dB F1':>8} | {'CPU(ms)':>8} | {'GPU(ms)':>8}")
print("-" * 105)
for r, pm, ca, cf, na, nf, tc, tg in rows:
    print(f"{r:<28} | {pm:>6.2f}M | {ca:>7.2f}% | {cf:>7.2f}% | {na:>7.2f}% | {nf:>7.2f}% | {tc:>7.2f} | {tg:>7.2f}")
