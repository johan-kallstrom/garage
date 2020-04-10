"""Baselines (value functions) which use NumPy as a numerical backend."""
from garage.torch.value_functions.gaussian_mlp_value_function import \
    GaussianMLPValueFunction

__all__ = ['GaussianMLPValueFunction']
