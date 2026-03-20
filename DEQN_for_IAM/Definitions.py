import importlib
import sys
import numpy as np
import tensorflow as tf
from Parameters import definitions, definition_bounds_hard, MODEL_NAME

# Dynamically import model-specific Definitions module
Model_Definitions = importlib.import_module(MODEL_NAME + ".Definitions")


def _make_clipped_def(name: str):
    """
    Returns f(s, ps) that computes Model_Definitions.<name>(s, ps) and clips
    the result to [lower, upper], with NumPy-2.0-safe infinities and
    dtype-safe bounds for TF.
    """
    fn = getattr(Model_Definitions, name)

    def f(s, ps):
        val = fn(s, ps)
        # Use -np.inf / np.inf (NumPy >= 2.0) instead of removed aliases
        lower = definition_bounds_hard["lower"].get(name, -np.inf)
        upper = definition_bounds_hard["upper"].get(name,  np.inf)
        # Ensure bounds match the dtype of 'val' for tf.clip_by_value
        lower = tf.cast(lower, val.dtype)
        upper = tf.cast(upper, val.dtype)
        return tf.clip_by_value(val, lower, upper)

    return f


def _make_passthrough_def(name: str):
    """Returns f(s, ps) == Model_Definitions.<name>(s, ps) (no clipping)."""
    fn = getattr(Model_Definitions, name)
    return lambda s, ps: fn(s, ps)


for d in definitions:
    if (d in definition_bounds_hard["lower"]) or (d in definition_bounds_hard["upper"]):
        accessor = _make_clipped_def(d)
    else:
        accessor = _make_passthrough_def(d)

    # Main accessor (possibly clipped)
    setattr(sys.modules[__name__], d, accessor)

    # Always provide a RAW (never clipped) variant — useful for penalties
    setattr(sys.modules[__name__], d + "_RAW", _make_passthrough_def(d))
