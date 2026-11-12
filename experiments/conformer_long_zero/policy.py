"""Declared budgets and a conservative, sustained overfitting indicator."""
import math
from statistics import mean

CONDITIONS = ("baseline", "speech_focus")
BATCHES = (32, 16)
LEARNING_RATES = (1e-6, 3e-6, 1e-5)
PILOT_EPOCHS = 5
MAX_EPOCH = 100


def update_count(epoch, batch_size):
    """The first ten epochs used batch32; continuation keeps609 draws/epoch."""
    return 200 + (epoch - 10) * math.ceil(609 / batch_size)


def select_learning_rate(rows):
    """Use all four folds' final three pilot epochs, not a lucky single peak."""
    candidates = []
    for lr in LEARNING_RATES:
        subset = [r for r in rows if r["encoder_lr"] == lr and r["epoch"] >= 13]
        assert len(subset) == 12, (lr, len(subset))
        candidates.append(dict(encoder_lr=lr,
            score=mean(r["zero_f1"] for r in subset),
            loss=mean(r["zero_loss"] for r in subset)))
    return min(candidates, key=lambda r: (-r["score"], r["loss"], r["encoder_lr"]))


def overfit_status(history):
    """Do not equate a plateau with overfitting, or wait forever for overfitting.

    Three-epoch averages must show lower training loss, higher fixed0dB loss,
    and worse0dB F1 for five successive observations. Record another five
    epochs after confirmation. These are operational indicators, not a proof.
    """
    smooth = []
    for i in range(2, len(history)):
        window = history[i-2:i+1]
        if any("train_loss" not in r for r in window):
            continue
        smooth.append(dict(epoch=window[-1]["epoch"], **{
            k: mean(r[k] for r in window)
            for k in ("train_loss", "zero_loss", "zero_f1")}))
    streak, confirmation, evidence = 0, None, None
    for current in smooth:
        previous = [r for r in smooth if r["epoch"] <= current["epoch"]-5]
        if not previous or current["epoch"] < 25:
            continue
        reference = min(previous, key=lambda r: r["zero_loss"])
        bad = (current["train_loss"] <= reference["train_loss"]*.98
               and current["zero_loss"] >= reference["zero_loss"]*1.03
               and current["zero_f1"] <= max(r["zero_f1"] for r in previous)-.005)
        streak = streak+1 if bad else 0
        if confirmation is None and streak >= 5:
            confirmation = current["epoch"]
            evidence = dict(reference=reference, current=current)
    epoch = history[-1]["epoch"]
    stop = confirmation is not None and epoch >= confirmation+5
    reason = "sustained_overfit_indicator_plus_five_epochs" if stop else None
    if epoch >= MAX_EPOCH:
        stop, reason = True, "maximum_epoch_reached"
    return dict(stop=stop, reason=reason, confirmation_epoch=confirmation,
                evidence=evidence, latest_bad_streak=streak)
