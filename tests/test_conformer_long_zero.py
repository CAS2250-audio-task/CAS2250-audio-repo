"""Stopping must distinguish sustained degradation from a validation plateau."""
from experiments.conformer_long_zero.policy import overfit_status, update_count, select_learning_rate


def test_plateau_is_not_reported_as_overfitting():
    history = [dict(epoch=e, train_loss=1/e, zero_loss=.8, zero_f1=.82)
               for e in range(10, 65)]
    result = overfit_status(history)
    assert not result["stop"] and result["confirmation_epoch"] is None


def test_persistent_train_validation_divergence_gets_a_followup_window():
    history = [dict(epoch=e, train_loss=1/e,
        zero_loss=1-e*.005 if e<=25 else .875+(e-25)*.02,
        zero_f1=.8+e*.001 if e<=25 else .825-(e-25)*.003)
        for e in range(11, 51)]
    observed = [overfit_status(history[:i]) for i in range(1, len(history)+1)]
    first = next(r for r in observed if r["confirmation_epoch"] is not None)
    confirmation = first["confirmation_epoch"]
    assert confirmation >= 25 and not first["stop"]
    assert not overfit_status([r for r in history if r["epoch"] < confirmation+5])["stop"]
    assert overfit_status([r for r in history if r["epoch"] <= confirmation+5])["stop"]


def test_cap_without_degradation_does_not_claim_overfit():
    history=[dict(epoch=e,train_loss=1/e,zero_loss=1/e,zero_f1=.9) for e in range(11,101)]
    result=overfit_status(history)
    assert result["stop"] and result["reason"]=="maximum_epoch_reached"
    assert result["confirmation_epoch"] is None
    assert update_count(15,32)==300 and update_count(15,16)==395


def test_lr_selection_uses_all_folds_and_last_three_epochs():
    rows=[dict(encoder_lr=lr,epoch=e,zero_f1=.9 if lr==3e-6 else .8,zero_loss=.5)
          for lr in (1e-6,3e-6,1e-5) for fold in range(4) for e in range(13,16)]
    assert select_learning_rate(rows)["encoder_lr"]==3e-6
