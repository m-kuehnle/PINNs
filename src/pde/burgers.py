"""
Physics-Informed Neural Network for 1D Burgers' Equation.

This module implements a PINN solver for the viscous Burgers' equation:
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

with initial condition:
    u(x, 0) = -sin(πx),  x ∈ [-1, 1]

and boundary conditions:
    u(-1, t) = u(1, t) = 0,  t ∈ [0, T]

where ν is the kinematic viscosity.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import numpy as np


class BurgersPINN:
    """
    Physics-Informed Neural Network for solving the 1D Burgers' equation.
    
    The PINN approach embeds the PDE, initial conditions (IC), and boundary 
    conditions (BC) directly into the loss function. The network learns to
    approximate the solution by minimizing:
    
        Loss = w_pde · Loss_PDE + w_ic · Loss_IC + w_bc · Loss_BC
    
    where:
        - Loss_PDE: Mean squared residual of the PDE
        - Loss_IC: Mean squared error at initial condition
        - Loss_BC: Mean squared error at boundaries
    
    Attributes:
        network (nn.Module): Neural network approximating u(x, t).
        nu (float): Kinematic viscosity parameter.
        device (torch.device): Computation device.
        
    Args:
        network: Neural network (takes [x, t] as input, outputs u).
        nu: Kinematic viscosity (default: 0.01/π).
        device: Computation device (optional).
        
    Example:
        >>> from src.pinn.network import MLP
        >>> network = MLP([2, 50, 50, 50, 1])
        >>> pinn = BurgersPINN(network, nu=0.01/np.pi)
        >>> loss = pinn.compute_loss(x_pde, t_pde, x_ic, t_ic, u_ic, x_bc, t_bc)
    """
    
    def __init__(
        self,
        network: nn.Module,
        nu: float = 0.01 / np.pi,
        device: torch.device = None
    ) -> None:
        """
        Initialize the Burgers' equation PINN.
        
        Args:
            network: Neural network for solution approximation.
            nu: Kinematic viscosity coefficient.
            device: Device for computation.
        """
        self.network = network
        self.nu = nu
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')  # Apple Silicon GPU
        else:
            self.device = torch.device('cpu')
        self.network.to(self.device)
    
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict solution u(x, t) using the trained network.
        
        Args:
            x: Spatial coordinates, shape (N,) or (N, 1).
            t: Temporal coordinates, shape (N,) or (N, 1).
        
        Returns:
            Predicted solution u(x, t), shape (N, 1).
        """
        # Ensure proper shape
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Concatenate inputs
        inputs = torch.cat([x, t], dim=1)
        inputs = inputs.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            u = self.network(inputs)
        
        return u
    
    def _compute_pde_residual(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PDE residual using automatic differentiation.
        
        The Burgers' equation residual is:
            R = ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x²
        
        Args:
            x: Spatial collocation points, shape (N, 1).
            t: Temporal collocation points, shape (N, 1).
        
        Returns:
            PDE residual at each collocation point, shape (N, 1).
        """
        # Enable gradient computation
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        
        # Forward pass
        inputs = torch.cat([x, t], dim=1)
        u = self.network(inputs)
        
        # Compute first-order derivatives using autograd
        # ∂u/∂t
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ∂u/∂x
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute second-order derivative ∂²u/∂x²
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Burgers' equation residual: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x²
        residual = u_t + u * u_x - self.nu * u_xx
        
        return residual
    
    def compute_loss(
        self,
        x_pde: torch.Tensor,
        t_pde: torch.Tensor,
        x_ic: torch.Tensor,
        t_ic: torch.Tensor,
        u_ic: torch.Tensor,
        x_bc: torch.Tensor,
        t_bc: torch.Tensor,
        w_pde: float = 1.0,
        w_ic: float = 1.0,
        w_bc: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-informed loss.
        
        Args:
            x_pde: Spatial collocation points for PDE, shape (N_pde, 1).
            t_pde: Temporal collocation points for PDE, shape (N_pde, 1).
            x_ic: Spatial points for initial condition, shape (N_ic, 1).
            t_ic: Temporal points for initial condition (all zeros), shape (N_ic, 1).
            u_ic: True values at initial condition, shape (N_ic, 1).
            x_bc: Spatial points for boundary conditions, shape (N_bc, 1).
            t_bc: Temporal points for boundary conditions, shape (N_bc, 1).
            w_pde: Weight for PDE loss component.
            w_ic: Weight for initial condition loss.
            w_bc: Weight for boundary condition loss.
        
        Returns:
            Tuple containing:
                - total_loss: Weighted sum of all loss components.
                - loss_dict: Dictionary with individual loss components.
        """
        # PDE residual loss
        residual = self._compute_pde_residual(x_pde, t_pde)
        loss_pde = torch.mean(residual ** 2)
        
        # Initial condition loss
        inputs_ic = torch.cat([x_ic, t_ic], dim=1)
        u_pred_ic = self.network(inputs_ic)
        loss_ic = torch.mean((u_pred_ic - u_ic) ** 2)
        
        # Boundary condition loss
        inputs_bc = torch.cat([x_bc, t_bc], dim=1)
        u_pred_bc = self.network(inputs_bc)
        u_bc = torch.zeros_like(u_pred_bc)  # Dirichlet BC: u = 0
        loss_bc = torch.mean((u_pred_bc - u_bc) ** 2)
        
        # Total weighted loss
        total_loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
        
        # Loss components for monitoring
        loss_dict = {
            'pde': loss_pde.item(),
            'ic': loss_ic.item(),
            'bc': loss_bc.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def evaluate_solution(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the solution on a 2D grid for visualization.
        
        Args:
            x_grid: Spatial grid, shape (N_x, N_t).
            t_grid: Temporal grid, shape (N_x, N_t).
        
        Returns:
            Solution u(x, t) evaluated on the grid, shape (N_x, N_t).
        """
        # Flatten grids
        x_flat = torch.tensor(x_grid.flatten(), dtype=torch.float32).unsqueeze(1)
        t_flat = torch.tensor(t_grid.flatten(), dtype=torch.float32).unsqueeze(1)
        
        # Predict
        u_flat = self.predict(x_flat, t_flat)
        
        # Reshape to grid
        u_grid = u_flat.cpu().numpy().reshape(x_grid.shape)
        
        return u_grid
