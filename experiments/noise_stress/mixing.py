"""Deterministic contiguous crops and additive SNR-controlled mixing."""
import hashlib

import numpy as np

SAMPLE_RATE = 16000
SAMPLES = 24000
LEVELS = (10, 0, -5, -10)
DRAWS = 3
SEED = 20261112


def crop_noise(recording, key, noise_id, draw):
    if recording.ndim != 1 or len(recording) < SAMPLES:
        raise ValueError("Expected a finite long mono recording")
    token = f"{SEED}|{noise_id}|{key}|{draw}".encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    for attempt in range(1000):
        start = int(rng.integers(0, len(recording) - SAMPLES + 1))
        noise = recording[start:start + SAMPLES].astype(np.float64)
        if not np.isfinite(noise).all():
            raise ValueError("Nonfinite noise window")
        noise -= noise.mean()  # Remove microphone DC before measuring noise power.
        power = float(np.mean(noise ** 2))
        if power >= 1e-8:  # Reject silent/near-silent windows (not a speech-content filter).
            return noise, dict(start_sample=start, end_sample=start+SAMPLES,
                               start_seconds=start/SAMPLE_RATE, draw=int(draw),
                               crop_seed=str(seed), crop_retry=attempt, noise_power=power)
    raise ValueError("No energetic noise window found")


def mix_at_snr(wave, noise, snr_db, reference_power):
    if wave.shape != noise.shape or not np.isfinite(wave).all() or not np.isfinite(noise).all():
        raise ValueError("Expected matching finite speech/noise arrays")
    noise_power = float(np.mean(np.asarray(noise, dtype=np.float64) ** 2))
    if reference_power <= 0 or noise_power <= 0:
        raise ValueError("SNR mixing needs positive reference and noise power")
    gain = float(np.sqrt(reference_power / (noise_power * 10 ** (snr_db/10))))
    mixed = wave.astype(np.float64) + gain * noise
    peak = float(np.max(np.abs(mixed)))
    attenuation = min(1.0, .99 / peak) if peak else 1.0
    mixed = (mixed * attenuation).astype(np.float32)
    measured = float(10*np.log10(reference_power / (noise_power * gain**2)))
    assert np.max(np.abs(mixed)) <= .990001 and abs(measured-snr_db) < 1e-8
    return mixed, dict(snr_db=snr_db, reference_power=reference_power, noise_gain=gain,
                       joint_attenuation=attenuation, snr_db_measured=measured,
                       mixture_sha256=hashlib.sha256(mixed.tobytes()).hexdigest())
