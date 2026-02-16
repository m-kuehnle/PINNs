#!/usr/bin/env python3
"""
Physics-Informed Neural Network for 1D Burgers' Equation

This script demonstrates solving the viscous Burgers' equation using PINNs:
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

with initial condition:
    u(x, 0) = -sin(πx),  x ∈ [-1, 1]

and boundary conditions:
    u(-1, t) = u(1, t) = 0,  t ∈ [0, 1]

Author: Senior Research Engineer for Scientific Machine Learning
Date: February 16, 2026
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pinn.network import MLP
from src.pinn.trainer import PINNTrainer
from src.pde.burgers import BurgersPINN
from src.utils.data import generate_training_data, generate_test_grid, initial_condition
from src.utils.visualization import plot_all


def main() -> None:
    """
    Main function to train and evaluate the PINN for Burgers' equation.
    """
    # ==================== Configuration ====================
    print("=" * 80)
    print(" Physics-Informed Neural Network for 1D Burgers' Equation")
    print("=" * 80)
    
    # Domain parameters
    X_MIN, X_MAX = -1.0, 1.0
    T_MIN, T_MAX = 0.0, 1.0
    NU = 0.01 / np.pi  # Kinematic viscosity
    
    # Network architecture
    LAYER_SIZES = [2, 50, 50, 50, 1]  # Input: (x, t), Output: u
    
    # Training parameters
    N_PDE = 10000      # Collocation points for PDE residual
    N_IC = 100         # Initial condition points
    N_BC = 100         # Boundary condition points (per boundary)
    ADAM_ITER = 10000  # Adam optimizer iterations
    LBFGS_ITER = 1000  # L-BFGS optimizer iterations
    ADAM_LR = 1e-3     # Adam learning rate
    
    # Loss weights
    W_PDE = 1.0
    W_IC = 1.0
    W_BC = 1.0
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nDevice: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\nDevice: {device}")
        print(f"GPU: Apple Silicon (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print(f"\nDevice: {device}")
        print("Note: No GPU acceleration available")
    
    # ==================== Network Initialization ====================
    print("\n" + "=" * 80)
    print("Initializing Neural Network")
    print("=" * 80)
    
    network = MLP(layer_sizes=LAYER_SIZES)
    print(f"Network Architecture: {LAYER_SIZES}")
    print(f"Total Parameters: {network.count_parameters():,}")
    print(f"Activation Function: tanh")
    print(f"Initialization: Xavier Uniform")
    
    # ==================== PINN Setup ====================
    print("\n" + "=" * 80)
    print("Setting up PINN Solver")
    print("=" * 80)
    
    pinn = BurgersPINN(network=network, nu=NU, device=device)
    print(f"PDE: Burgers' Equation")
    print(f"Viscosity (ν): {NU:.6f}")
    print(f"Domain: x ∈ [{X_MIN}, {X_MAX}], t ∈ [{T_MIN}, {T_MAX}]")
    print(f"IC: u(x, 0) = -sin(πx)")
    print(f"BC: u({X_MIN}, t) = u({X_MAX}, t) = 0")
    
    # ==================== Data Generation ====================
    print("\n" + "=" * 80)
    print("Generating Training Data")
    print("=" * 80)
    
    data = generate_training_data(
        n_pde=N_PDE,
        n_ic=N_IC,
        n_bc=N_BC,
        x_min=X_MIN,
        x_max=X_MAX,
        t_min=T_MIN,
        t_max=T_MAX,
        device=device
    )
    
    print(f"PDE collocation points: {N_PDE}")
    print(f"Initial condition points: {N_IC}")
    print(f"Boundary condition points: {2 * N_BC} (2 boundaries)")
    print(f"Total training points: {N_PDE + N_IC + 2 * N_BC}")
    
    # ==================== Training Setup ====================
    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    
    print(f"Optimizer Stage 1: Adam (iterations: {ADAM_ITER}, lr: {ADAM_LR})")
    print(f"Optimizer Stage 2: L-BFGS (max iterations: {LBFGS_ITER})")
    print(f"Loss weights: PDE={W_PDE}, IC={W_IC}, BC={W_BC}")
    
    # Define loss function closure
    def loss_fn() -> torch.Tensor:
        """Compute total physics-informed loss."""
        total_loss, _ = pinn.compute_loss(
            x_pde=data['x_pde'],
            t_pde=data['t_pde'],
            x_ic=data['x_ic'],
            t_ic=data['t_ic'],
            u_ic=data['u_ic'],
            x_bc=data['x_bc'],
            t_bc=data['t_bc'],
            w_pde=W_PDE,
            w_ic=W_IC,
            w_bc=W_BC
        )
        return total_loss
    
    # ==================== Training ====================
    trainer = PINNTrainer(network=network, device=device)
    
    history = trainer.train(
        loss_fn=loss_fn,
        adam_iterations=ADAM_ITER,
        lbfgs_iterations=LBFGS_ITER,
        adam_lr=ADAM_LR,
        print_every=1000
    )
    
    # ==================== Evaluation ====================
    print("\n" + "=" * 80)
    print("Evaluating Solution")
    print("=" * 80)
    
    # Generate test grid
    X, T = generate_test_grid(n_x=200, n_t=200, x_min=X_MIN, x_max=X_MAX, t_min=T_MIN, t_max=T_MAX)
    
    # Evaluate solution on grid
    print("Computing solution on 200×200 grid...")
    U = pinn.evaluate_solution(X, T)
    print("Solution evaluation complete!")
    
    # Extract 1D slices for comparison
    x_1d = X[:, 0]  # Spatial coordinates
    u_initial = initial_condition(x_1d)  # IC: u(x, 0)
    u_final = U[:, -1]  # Final state: u(x, T)
    
    # ==================== Visualization ====================
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate all plots
    fig_3d, fig_comp, fig_loss = plot_all(
        X=X,
        T=T,
        U=U,
        x_1d=x_1d,
        u_initial=u_initial,
        u_final=u_final,
        loss_history=history['total_loss'],
        adam_iterations=ADAM_ITER,
        save_dir=str(output_dir),
        show=False  # Set to True to display plots interactively
    )
    
    print("\nVisualization complete!")
    print(f"  - 3D Surface Plot: {output_dir}/solution_3d.png")
    print(f"  - Initial vs Final: {output_dir}/comparison.png")
    print(f"  - Loss Evolution: {output_dir}/loss_history.png")
    
    # ==================== Final Summary ====================
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    
    final_loss = history['total_loss'][-1]
    min_loss = min(history['total_loss'])
    
    print(f"Final Loss: {final_loss:.6e}")
    print(f"Minimum Loss: {min_loss:.6e}")
    print(f"Total Iterations: {len(history['total_loss'])}")
    
    # Compute residuals for diagnostics
    with torch.no_grad():
        _, loss_components = pinn.compute_loss(
            x_pde=data['x_pde'],
            t_pde=data['t_pde'],
            x_ic=data['x_ic'],
            t_ic=data['t_ic'],
            u_ic=data['u_ic'],
            x_bc=data['x_bc'],
            t_bc=data['t_bc']
        )
    
    print("\nLoss Components:")
    print(f"  - PDE Residual: {loss_components['pde']:.6e}")
    print(f"  - Initial Condition: {loss_components['ic']:.6e}")
    print(f"  - Boundary Condition: {loss_components['bc']:.6e}")
    
    # ==================== Save Model ====================
    model_path = output_dir / "burgers_pinn_model.pt"
    trainer.save_checkpoint(str(model_path))
    
    print("\n" + "=" * 80)
    print("PINN Training Complete!")
    print("=" * 80)
    print(f"\nModel saved to: {model_path}")
    print(f"Results saved to: {output_dir}/")
    print("\nThank you for using PINN for Burgers' Equation!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run main function
    main()
