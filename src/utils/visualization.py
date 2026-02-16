"""
Visualization Utilities for PINN Solutions.

This module provides high-quality visualization functions for:
- 3D surface plots of solutions
- Initial vs. final state comparisons
- Loss evolution during training
"""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


def plot_solution(
    X: np.ndarray,
    T: np.ndarray,
    U: np.ndarray,
    title: str = "PINN Solution: 1D Burgers' Equation",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a high-quality 3D surface plot of the solution u(x, t).
    
    Args:
        X: Spatial grid, shape (N_x, N_t).
        T: Temporal grid, shape (N_x, N_t).
        U: Solution values, shape (N_x, N_t).
        title: Plot title.
        save_path: Path to save the figure (optional).
        show: Whether to display the plot.
    
    Returns:
        Matplotlib figure object.
    
    Example:
        >>> X, T = generate_test_grid(200, 200)
        >>> U = pinn.evaluate_solution(X, T)
        >>> fig = plot_solution(X, T, U)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(
        X, T, U,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        edgecolor='none'
    )
    
    # Customize axes
    ax.set_xlabel('Space (x)', fontsize=12, labelpad=10)
    ax.set_ylabel('Time (t)', fontsize=12, labelpad=10)
    ax.set_zlabel('Solution u(x, t)', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('u(x, t)', fontsize=11)
    
    # Set viewing angle for better visualization
    ax.view_init(elev=25, azim=45)
    
    # Improve layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"3D surface plot saved to {save_path}")
    
    # Show plot
    if show:
        plt.show()
    
    return fig


def plot_comparison(
    x: np.ndarray,
    u_initial: np.ndarray,
    u_final: np.ndarray,
    t_final: float = 1.0,
    title: str = "Initial vs Final State",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a side-by-side comparison of initial and final states.
    
    Args:
        x: Spatial coordinates, shape (N,).
        u_initial: Initial condition u(x, 0), shape (N,).
        u_final: Final state u(x, T), shape (N,).
        t_final: Final time value for labeling.
        title: Plot title.
        save_path: Path to save the figure (optional).
        show: Whether to display the plot.
    
    Returns:
        Matplotlib figure object.
    
    Example:
        >>> x = np.linspace(-1, 1, 200)
        >>> u_initial = -np.sin(np.pi * x)
        >>> u_final = pinn.predict(torch.tensor(x), torch.tensor([1.0]*200))
        >>> fig = plot_comparison(x, u_initial, u_final)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot initial condition
    axes[0].plot(x, u_initial, 'b-', linewidth=2, label='u(x, 0)')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    axes[0].set_xlabel('Space (x)', fontsize=12)
    axes[0].set_ylabel('u(x, t)', fontsize=12)
    axes[0].set_title('Initial Condition (t = 0)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim([x.min(), x.max()])
    
    # Plot final state
    axes[1].plot(x, u_final, 'r-', linewidth=2, label=f'u(x, {t_final})')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].set_xlabel('Space (x)', fontsize=12)
    axes[1].set_ylabel('u(x, t)', fontsize=12)
    axes[1].set_title(f'Final State (t = {t_final})', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].set_xlim([x.min(), x.max()])
    
    # Overall title
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Comparison plot saved to {save_path}")
    
    # Show plot
    if show:
        plt.show()
    
    return fig


def plot_loss_history(
    loss_history: List[float],
    title: str = "Loss Evolution During Training",
    log_scale: bool = True,
    adam_iterations: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot the training loss evolution over iterations.
    
    Args:
        loss_history: List of loss values at each iteration.
        title: Plot title.
        log_scale: Whether to use logarithmic scale for y-axis.
        adam_iterations: Number of Adam iterations (marks transition to L-BFGS).
        save_path: Path to save the figure (optional).
        show: Whether to display the plot.
    
    Returns:
        Matplotlib figure object.
    
    Example:
        >>> history = trainer.train(loss_fn)
        >>> fig = plot_loss_history(history['total_loss'], adam_iterations=10000)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations = np.arange(len(loss_history))
    
    # Plot loss curve
    ax.plot(iterations, loss_history, 'b-', linewidth=1.5, alpha=0.8, label='Total Loss')
    
    # Mark transition from Adam to L-BFGS
    if adam_iterations and adam_iterations < len(loss_history):
        ax.axvline(
            x=adam_iterations,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label='Adam → L-BFGS'
        )
        
        # Add annotations
        ax.text(
            adam_iterations * 0.5,
            ax.get_ylim()[1] * 0.8,
            'Adam Phase',
            fontsize=11,
            ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )
        ax.text(
            adam_iterations + (len(loss_history) - adam_iterations) * 0.5,
            ax.get_ylim()[1] * 0.8,
            'L-BFGS Phase',
            fontsize=11,
            ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5)
        )
    
    # Set logarithmic scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Labels and title
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss (log scale)' if log_scale else 'Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11)
    
    # Add statistics box
    final_loss = loss_history[-1]
    min_loss = min(loss_history)
    stats_text = f'Final Loss: {final_loss:.2e}\nMin Loss: {min_loss:.2e}'
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Loss history plot saved to {save_path}")
    
    # Show plot
    if show:
        plt.show()
    
    return fig


def plot_all(
    X: np.ndarray,
    T: np.ndarray,
    U: np.ndarray,
    x_1d: np.ndarray,
    u_initial: np.ndarray,
    u_final: np.ndarray,
    loss_history: List[float],
    adam_iterations: Optional[int] = None,
    save_dir: Optional[str] = None,
    show: bool = True
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Create all three visualizations at once.
    
    Args:
        X: Spatial grid for 3D plot, shape (N_x, N_t).
        T: Temporal grid for 3D plot, shape (N_x, N_t).
        U: Solution for 3D plot, shape (N_x, N_t).
        x_1d: Spatial coordinates for 1D comparison, shape (N,).
        u_initial: Initial condition for comparison, shape (N,).
        u_final: Final state for comparison, shape (N,).
        loss_history: List of loss values.
        adam_iterations: Number of Adam iterations.
        save_dir: Directory to save figures (optional).
        show: Whether to display plots.
    
    Returns:
        Tuple of (fig_3d, fig_comparison, fig_loss).
    """
    # 3D surface plot
    save_path_3d = f"{save_dir}/solution_3d.png" if save_dir else None
    fig_3d = plot_solution(X, T, U, save_path=save_path_3d, show=show)
    
    # Initial vs final comparison
    save_path_comp = f"{save_dir}/comparison.png" if save_dir else None
    fig_comp = plot_comparison(
        x_1d, u_initial, u_final,
        save_path=save_path_comp,
        show=show
    )
    
    # Loss history
    save_path_loss = f"{save_dir}/loss_history.png" if save_dir else None
    fig_loss = plot_loss_history(
        loss_history,
        adam_iterations=adam_iterations,
        save_path=save_path_loss,
        show=show
    )
    
    return fig_3d, fig_comp, fig_loss
