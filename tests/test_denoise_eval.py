"""Guard mixture-only adapters against input mutation and invalid windows."""
import inspect

import numpy as np
import pytest

from experiments.denoise_eval.filters import Identity, Filter


def test_identity_preserves_gain_above_unit_range_without_mutating_input():
    x = np.linspace(-1.2, 1.2, 48000, dtype=np.float32).reshape(2, 24000)
    before = x.copy()
    y = Identity()(x)
    assert np.array_equal(x, before) and np.array_equal(y, x)
    y[0] = 0
    assert np.array_equal(x, before)
    assert list(inspect.signature(Filter.__call__).parameters) == ["self", "mixtures"]


def test_adapter_rejects_wrong_lengths_dtype_and_nonfinite_input():
    for bad in (np.zeros((1, 23999), np.float32), np.zeros((1,24000), np.float64),
                np.full((1,24000), np.nan, np.float32), np.zeros(24000, np.float32)):
        with pytest.raises(ValueError):
            Identity()(bad)


def test_adapter_rejects_nonfinite_or_changed_length_output():
    class Invalid(Filter):
        def process(self, x):
            return x[:, :-1]
    with pytest.raises(ValueError):
        Invalid()(np.zeros((1,24000),np.float32))
