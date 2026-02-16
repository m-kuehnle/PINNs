"""
Core PINN module containing neural network architectures and training utilities.
"""

from .network import MLP
from .trainer import PINNTrainer

__all__ = ["MLP", "PINNTrainer"]
