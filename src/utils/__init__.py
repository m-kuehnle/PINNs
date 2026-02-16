"""
Utility functions for data generation and visualization.
"""

from .data import generate_training_data
from .visualization import plot_solution, plot_loss_history, plot_comparison

__all__ = ["generate_training_data", "plot_solution", "plot_loss_history", "plot_comparison"]
