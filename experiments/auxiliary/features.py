"""Deterministic WAV-only descriptors, missingness, and train-only scaling."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    samples: int = 24000
    hop: int = 160
    n_fft: int = 512
    pitch_frame: int = 1024
    pitch_min: float = 65.0
    pitch_max: float = 600.0
    formant_ceiling: float = 5500.0
    formant_window: float = 0.025
    delta_radius: int = 2
    version: str = "aligned-acoustics-v2-support"

    def to_dict(self):
        return asdict(self)


NAMES = ["log_f0", "voiced", "voiced_prob", "f1", "f2", "f3",
         "b1", "b2", "b3", "delta_log_f0", "delta_f1", "delta_f2", "delta_f3",
         "flatness", "flux", "centroid", "log_rms"]
SETS = {
    "0": [],
    "1": ["log_f0", "delta_log_f0", "voiced", "voiced_prob"],
    "2": ["f1", "f2", "f3"],
    "3": ["f1", "f2", "f3", "b1", "b2", "b3"],
    "4": ["f1", "f2", "f3", "delta_f1", "delta_f2", "delta_f3"],
    "5": ["flatness", "flux", "centroid"],
    "6": ["log_rms"],
    "7": NAMES.copy(),
}
BINARY = {"voiced", "voiced_prob"}


def masked_delta(values, valid, hop_seconds, radius=2):
    """Local slope only where the complete window is observed; never bridge a gap."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    positions = np.arange(-radius, radius + 1, dtype=float) * hop_seconds
    denom = np.sum(positions ** 2)
    for t in range(radius, len(values) - radius):
        sl = slice(t - radius, t + radius + 1)
        if valid[sl].all():
            out[t] = np.dot(positions, values[sl]) / denom
    return out


def apply_support(wav, raw, cfg=FeatureConfig()):
    """Reject digitally empty analysis windows, not quiet/unvoiced speech.

    LPC can manufacture finite roots in a silent span of an otherwise non-silent
    recording. Availability must therefore also reflect the waveform support.
    This is not VAD and does not establish correctness of a nonempty estimate.
    """
    import librosa
    raw = np.asarray(raw, dtype=np.float32).copy()
    idx = {n: i for i, n in enumerate(NAMES)}
    def supported(length):
        rms = librosa.feature.rms(y=np.asarray(wav, dtype=np.float64), frame_length=length,
                                 hop_length=cfg.hop, center=True, pad_mode="constant")[0]
        return rms > 1e-8
    pitch_support = supported(cfg.pitch_frame)
    raw[idx["log_f0"], ~pitch_support] = np.nan
    raw[idx["voiced"], ~pitch_support] = 0
    raw[idx["voiced_prob"], ~pitch_support] = 0
    # Praat's Gaussian window actual length is twice its effective window length.
    formant_support = supported(round(2*cfg.formant_window*cfg.sample_rate))
    for name in ["f1","f2","f3","b1","b2","b3"]:
        raw[idx[name], ~formant_support] = np.nan
    for name in ["log_f0","f1","f2","f3"]:
        values = raw[idx[name]]
        raw[idx[f"delta_{name}"]] = masked_delta(
            values, np.isfinite(values), cfg.hop/cfg.sample_rate, cfg.delta_radius)
    return raw, np.isfinite(raw)


def extract(wav, cfg=FeatureConfig()):
    """No labels, identities, VAD timestamps, or split information enter extraction."""
    import librosa
    import parselmouth

    wav = np.asarray(wav, dtype=np.float64)
    if wav.shape != (cfg.samples,) or not np.isfinite(wav).all():
        raise ValueError("Expected finite, mono, fixed-length waveform")
    frames = cfg.samples // cfg.hop + 1
    times = np.arange(frames) * cfg.hop / cfg.sample_rate
    raw = np.full((len(NAMES), frames), np.nan, dtype=np.float64)
    idx = {n: i for i, n in enumerate(NAMES)}

    # Silent/constant signals have no meaningful F0. Avoid a costly pitch search.
    if np.max(np.abs(wav)) > 1e-8 and np.std(wav) > 1e-8:
        f0, voiced, prob = librosa.pyin(
            wav, sr=cfg.sample_rate, fmin=cfg.pitch_min, fmax=cfg.pitch_max,
            frame_length=cfg.pitch_frame, hop_length=cfg.hop, center=True,
            fill_na=np.nan)
        if len(f0) != frames:
            raise ValueError("Pitch frame centres do not match the common time grid")
        ok = np.isfinite(f0) & (f0 > 0) & voiced
        raw[idx["log_f0"], ok] = 12 * np.log2(f0[ok] / 100.0)
        raw[idx["voiced"]] = voiced.astype(float)
        raw[idx["voiced_prob"]] = prob
        formants = parselmouth.Sound(wav, cfg.sample_rate).to_formant_burg(
            time_step=cfg.hop / cfg.sample_rate, max_number_of_formants=5,
            maximum_formant=cfg.formant_ceiling, window_length=cfg.formant_window,
            pre_emphasis_from=50.0)
        for k in (1, 2, 3):
            f = np.array([formants.get_value_at_time(k, float(t)) for t in times])
            b = np.array([formants.get_bandwidth_at_time(k, float(t)) for t in times])
            # Availability checks, not a claim that the formant assignment is correct.
            f_ok = np.isfinite(f) & (f > 50) & (f < cfg.formant_ceiling - 50)
            b_ok = f_ok & np.isfinite(b) & (b > 0)
            raw[idx[f"f{k}"], f_ok] = f[f_ok]
            raw[idx[f"b{k}"], b_ok] = b[b_ok]
    else:
        raw[idx["voiced"]] = 0
        raw[idx["voiced_prob"]] = 0

    spectrum = np.abs(librosa.stft(wav, n_fft=cfg.n_fft, win_length=cfg.n_fft,
                                  hop_length=cfg.hop, center=True, pad_mode="reflect"))
    power = spectrum ** 2
    energy = power.sum(axis=0)
    support = energy > 1e-12
    frequencies = np.fft.rfftfreq(cfg.n_fft, 1 / cfg.sample_rate)
    centroid = np.full(frames, np.nan)
    centroid[support] = (spectrum[:, support] * frequencies[:, None]).sum(0) / spectrum[:, support].sum(0)
    raw[idx["centroid"]] = centroid
    # A fixed floor is used only to evaluate the geometric mean, not to label silence noise.
    flatness = np.exp(np.mean(np.log(np.maximum(power, 1e-12)), axis=0)) / np.maximum(power.mean(0), 1e-12)
    raw[idx["flatness"], support] = flatness[support]
    # L1-normalized magnitude flux: less sensitive to an overall gain change, not noise invariant.
    normalized = spectrum / np.maximum(spectrum.sum(0, keepdims=True), 1e-12)
    raw[idx["flux"], 1:] = np.sqrt(np.mean(np.diff(normalized, axis=1) ** 2, axis=0))
    rms = librosa.feature.rms(y=wav, frame_length=cfg.n_fft, hop_length=cfg.hop,
                             center=True, pad_mode="reflect")[0]
    raw[idx["log_rms"]] = 20 * np.log10(np.maximum(rms, 1e-8))
    return apply_support(wav, raw, cfg)


def fit_scaler(raw, valid):
    """raw/valid are N,D,T and MUST contain only training examples."""
    mean, std, count = [], [], []
    for j, name in enumerate(NAMES):
        values = raw[:, j, :][valid[:, j, :]]
        count.append(int(len(values)))
        if name in BINARY:
            mean.append(0.0)
            std.append(1.0)
        else:
            if not len(values):
                raise ValueError(f"No training measurements for {name}")
            mean.append(float(values.astype(np.float64).mean()))
            std.append(max(float(values.astype(np.float64).std()), 1e-6))
    return dict(names=NAMES, mean=mean, std=std, observed_frames=count,
                fitted_on="train_only_valid_frames")


def encode(raw, valid, scaler, set_id):
    """Zero imputation AFTER scaling; one availability channel per descriptor."""
    columns = [NAMES.index(n) for n in SETS[str(set_id)]]
    if not columns:
        return np.zeros((len(raw), 0, raw.shape[-1]), dtype=np.float32)
    observed = valid[:, columns, :]
    values = np.zeros_like(raw[:, columns, :], dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)[columns][None, :, None]
    std = np.asarray(scaler["std"], dtype=np.float32)[columns][None, :, None]
    # Safe finite replacement precedes arithmetic; NaN never reaches a neural layer.
    safe = np.where(observed, raw[:, columns, :], mean)
    np.subtract(safe, mean, out=values)
    values /= std
    values[~observed] = 0
    result = np.concatenate([values, observed.astype(np.float32)], axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite encoded auxiliary input")
    return result
