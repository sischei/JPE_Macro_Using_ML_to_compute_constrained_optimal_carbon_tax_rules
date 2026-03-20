#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:26:23 2020

@author: -
"""
import sys
import numpy as np
import tensorflow as tf
from Parameters import policy_states, policy_bounds_hard, policy


def _make_clipped_accessor(ind: int):
    """
    Returns a function f(x) that extracts column 'ind' from x and clips it
    to [lower, upper], where bounds come from policy_bounds_hard with safe defaults.
    Bounds are cast to the dtype of x for TF compatibility.
    """
    def accessor(x):
        # Defaults updated for NumPy 2.0+: use -np.inf / np.inf
        lower = policy_bounds_hard['lower'].get(policy_states[ind], -np.inf)
        upper = policy_bounds_hard['upper'].get(policy_states[ind],  np.inf)

        # Ensure bounds have the same dtype as x[:, ind]
        lower = tf.cast(lower, x.dtype)
        upper = tf.cast(upper, x.dtype)

        return tf.clip_by_value(x[:, ind], lower, upper)
    return accessor


def _make_passthrough_accessor(ind: int):
    """Returns a function f(x) that extracts column 'ind' from x without clipping."""
    return lambda x: x[:, ind]


def _make_policy_from_state(ind: int):
    """Returns a function f(state) that evaluates the policy and extracts column 'ind'."""
    return lambda state: policy(state)[:, ind]


for i, policy_state in enumerate(policy_states):
    if (policy_state in policy_bounds_hard['lower']) or (policy_state in policy_bounds_hard['upper']):
        accessor = _make_clipped_accessor(i)
    else:
        accessor = _make_passthrough_accessor(i)

    # Main accessor (possibly clipped)
    setattr(sys.modules[__name__], policy_state, accessor)

    # Always add a 'RAW' accessor (never clipped) — useful for penalties
    setattr(sys.modules[__name__], policy_state + "_RAW", _make_passthrough_accessor(i))

    # Policy function: compute from current state via current policy
    setattr(sys.modules[__name__], policy_state + "_POLICY_FROM_STATE", _make_policy_from_state(i))
