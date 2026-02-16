"""
Data Generation Utilities for PINN Training.

This module provides functions for generating training data including:
- Initial conditions (IC)
- Boundary conditions (BC)
- Collocation points for PDE residual
"""

from typing import Dict, Tuple
import torch
import numpy as np


def generate_training_data(
    n_pde: int = 10000,
    n_ic: int = 100,
    n_bc: int = 100,
    x_min: float = -1.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Generate training data for the 1D Burgers' equation PINN.
    
    This function creates three types of training points:
    1. Collocation points (x, t) in the interior domain for PDE residual
    2. Initial condition points at t = 0
    3. Boundary condition points at x = -1 and x = 1
    
    Args:
        n_pde: Number of collocation points for PDE residual.
        n_ic: Number of points for initial condition.
        n_bc: Number of points for each boundary (total 2*n_bc points).
        x_min: Minimum spatial coordinate.
        x_max: Maximum spatial coordinate.
        t_min: Minimum time (typically 0).
        t_max: Maximum time.
        device: Computation device (CPU or CUDA).
    
    Returns:
        Dictionary containing:
            - 'x_pde': Spatial collocation points, shape (n_pde, 1)
            - 't_pde': Temporal collocation points, shape (n_pde, 1)
            - 'x_ic': Spatial points for IC, shape (n_ic, 1)
            - 't_ic': Temporal points for IC (all zeros), shape (n_ic, 1)
            - 'u_ic': Initial condition values, shape (n_ic, 1)
            - 'x_bc': Spatial points for BC, shape (2*n_bc, 1)
            - 't_bc': Temporal points for BC, shape (2*n_bc, 1)
    
    Example:
        >>> data = generate_training_data(n_pde=10000, n_ic=100, n_bc=100)
        >>> x_pde, t_pde = data['x_pde'], data['t_pde']
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== PDE Collocation Points ====================
    # Random sampling in the spatiotemporal domain [x_min, x_max] × [t_min, t_max]
    x_pde = torch.rand(n_pde, 1, device=device) * (x_max - x_min) + x_min
    t_pde = torch.rand(n_pde, 1, device=device) * (t_max - t_min) + t_min
    
    # ==================== Initial Condition (t = 0) ====================
    # Uniform sampling along x at t = 0
    x_ic = torch.linspace(x_min, x_max, n_ic, device=device).unsqueeze(1)
    t_ic = torch.zeros(n_ic, 1, device=device)
    
    # Initial condition: u(x, 0) = -sin(πx)
    u_ic = -torch.sin(np.pi * x_ic)
    
    # ==================== Boundary Conditions ====================
    # Left boundary: x = x_min, t ∈ [0, T]
    t_bc_left = torch.rand(n_bc, 1, device=device) * (t_max - t_min) + t_min
    x_bc_left = torch.full((n_bc, 1), x_min, device=device)
    
    # Right boundary: x = x_max, t ∈ [0, T]
    t_bc_right = torch.rand(n_bc, 1, device=device) * (t_max - t_min) + t_min
    x_bc_right = torch.full((n_bc, 1), x_max, device=device)
    
    # Concatenate left and right boundaries
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
    t_bc = torch.cat([t_bc_left, t_bc_right], dim=0)
    
    # Return all training data
    return {
        'x_pde': x_pde,
        't_pde': t_pde,
        'x_ic': x_ic,
        't_ic': t_ic,
        'u_ic': u_ic,
        'x_bc': x_bc,
        't_bc': t_bc
    }


def generate_test_grid(
    n_x: int = 200,
    n_t: int = 200,
    x_min: float = -1.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a uniform 2D grid for solution evaluation and visualization.
    
    Args:
        n_x: Number of spatial grid points.
        n_t: Number of temporal grid points.
        x_min: Minimum spatial coordinate.
        x_max: Maximum spatial coordinate.
        t_min: Minimum time.
        t_max: Maximum time.
    
    Returns:
        Tuple of (X, T) meshgrid arrays, each of shape (n_x, n_t).
    
    Example:
        >>> X, T = generate_test_grid(n_x=200, n_t=200)
        >>> X.shape, T.shape
        ((200, 200), (200, 200))
    """
    x = np.linspace(x_min, x_max, n_x)
    t = np.linspace(t_min, t_max, n_t)
    X, T = np.meshgrid(x, t, indexing='ij')
    
    return X, T


def initial_condition(x: np.ndarray) -> np.ndarray:
    """
    Compute the initial condition for Burgers' equation.
    
    Args:
        x: Spatial coordinates, shape (N,) or (N, M).
    
    Returns:
        Initial condition u(x, 0) = -sin(πx), same shape as input.
    
    Example:
        >>> x = np.linspace(-1, 1, 100)
        >>> u0 = initial_condition(x)
    """
    return -np.sin(np.pi * x)


def sample_random_points(
    n_points: int,
    x_min: float = -1.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random (x, t) points in the spatiotemporal domain.
    
    Args:
        n_points: Number of random points to generate.
        x_min: Minimum spatial coordinate.
        x_max: Maximum spatial coordinate.
        t_min: Minimum time.
        t_max: Maximum time.
        device: Computation device.
    
    Returns:
        Tuple of (x, t) tensors, each of shape (n_points, 1).
    
    Example:
        >>> x, t = sample_random_points(1000)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')  # Apple Silicon GPU
        else:
            device = torch.device('cpu')
    
    x = torch.rand(n_points, 1, device=device) * (x_max - x_min) + x_min
    t = torch.rand(n_points, 1, device=device) * (t_max - t_min) + t_min
    
    return x, t
