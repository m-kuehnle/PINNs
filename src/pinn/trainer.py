"""
PINN Training Utilities with Two-Stage Optimization.

This module provides training functionality for Physics-Informed Neural Networks
using a two-stage optimization approach: Adam for initial training followed by
L-BFGS for fine-tuning to achieve high precision.
"""

from typing import Dict, List, Callable, Optional, Tuple
import torch
import torch.optim as optim
import numpy as np


class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Networks with two-stage optimization.
    
    This trainer implements a robust optimization strategy:
    1. Stage 1: Adam optimizer for rapid initial convergence
    2. Stage 2: L-BFGS optimizer for high-precision fine-tuning
    
    Attributes:
        network (torch.nn.Module): The neural network to train.
        device (torch.device): Device for computation (CPU or CUDA).
        loss_history (List[float]): History of loss values during training.
        
    Args:
        network: Neural network model (e.g., MLP).
        device: Computation device. If None, uses CUDA if available, else CPU.
        
    Example:
        >>> network = MLP([2, 50, 50, 1])
        >>> trainer = PINNTrainer(network)
        >>> loss_fn = lambda: compute_physics_loss()
        >>> trainer.train(loss_fn, adam_iterations=10000, lbfgs_iterations=1000)
    """
    
    def __init__(
        self,
        network: torch.nn.Module,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize the PINN trainer.
        
        Args:
            network: Neural network to train.
            device: Computation device (optional).
        """
        self.network = network
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')  # Apple Silicon GPU
        else:
            self.device = torch.device('cpu')
        self.network.to(self.device)
        
        self.loss_history: List[float] = []
        self._iteration: int = 0
    
    def train(
        self,
        loss_fn: Callable[[], torch.Tensor],
        adam_iterations: int = 10000,
        lbfgs_iterations: int = 1000,
        adam_lr: float = 1e-3,
        print_every: int = 1000,
        convergence_threshold: float = 1e-6
    ) -> Dict[str, List[float]]:
        """
        Train the network using two-stage optimization.
        
        Args:
            loss_fn: Callable that computes and returns the total loss.
                    Should be a closure that captures all necessary data.
            adam_iterations: Number of iterations for Adam optimizer.
            lbfgs_iterations: Maximum number of iterations for L-BFGS.
            adam_lr: Learning rate for Adam optimizer.
            print_every: Print progress every N iterations.
            convergence_threshold: Convergence criterion for early stopping.
        
        Returns:
            Dictionary containing training history with keys:
                - 'total_loss': List of total loss values
                - 'iterations': List of iteration numbers
        """
        self.loss_history = []
        self._iteration = 0
        
        print(f"Training on device: {self.device}")
        print(f"Network parameters: {self._count_parameters()}")
        print("=" * 70)
        
        # Stage 1: Adam optimization
        print("\nStage 1: Adam Optimization")
        print("-" * 70)
        self._train_adam(
            loss_fn=loss_fn,
            iterations=adam_iterations,
            lr=adam_lr,
            print_every=print_every
        )
        
        # Stage 2: L-BFGS optimization
        print("\n" + "=" * 70)
        print("Stage 2: L-BFGS Optimization (Fine-tuning)")
        print("-" * 70)
        self._train_lbfgs(
            loss_fn=loss_fn,
            max_iterations=lbfgs_iterations,
            print_every=print_every,
            convergence_threshold=convergence_threshold
        )
        
        print("\n" + "=" * 70)
        print(f"Training completed!")
        print(f"Final loss: {self.loss_history[-1]:.6e}")
        print(f"Total iterations: {self._iteration}")
        
        return {
            'total_loss': self.loss_history,
            'iterations': list(range(len(self.loss_history)))
        }
    
    def _train_adam(
        self,
        loss_fn: Callable[[], torch.Tensor],
        iterations: int,
        lr: float,
        print_every: int
    ) -> None:
        """
        Train using Adam optimizer.
        
        Args:
            loss_fn: Loss function callable.
            iterations: Number of training iterations.
            lr: Learning rate.
            print_every: Print frequency.
        """
        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        for i in range(iterations):
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            optimizer.step()
            
            # Record loss
            loss_value = loss.item()
            self.loss_history.append(loss_value)
            self._iteration += 1
            
            # Print progress
            if (i + 1) % print_every == 0 or i == 0:
                print(f"Iteration {i + 1:6d}/{iterations:6d} | Loss: {loss_value:.6e}")
    
    def _train_lbfgs(
        self,
        loss_fn: Callable[[], torch.Tensor],
        max_iterations: int,
        print_every: int,
        convergence_threshold: float
    ) -> None:
        """
        Train using L-BFGS optimizer for fine-tuning.
        
        L-BFGS is a quasi-Newton method that uses limited memory to approximate
        the Hessian matrix, providing faster convergence for well-conditioned problems.
        
        Args:
            loss_fn: Loss function callable.
            max_iterations: Maximum number of L-BFGS iterations.
            print_every: Print frequency.
            convergence_threshold: Threshold for convergence detection.
        """
        optimizer = optim.LBFGS(
            self.network.parameters(),
            lr=1.0,
            max_iter=20,  # Max iterations per step
            max_eval=25,  # Max function evaluations per step
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=100,
            line_search_fn='strong_wolfe'
        )
        
        iteration_count = 0
        previous_loss = float('inf')
        
        def closure() -> torch.Tensor:
            """Closure for L-BFGS optimizer."""
            nonlocal iteration_count, previous_loss
            
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            
            # Record loss
            loss_value = loss.item()
            self.loss_history.append(loss_value)
            self._iteration += 1
            iteration_count += 1
            
            # Print progress
            if iteration_count % print_every == 0 or iteration_count == 1:
                print(f"Iteration {iteration_count:6d}/{max_iterations:6d} | Loss: {loss_value:.6e}")
            
            # Check convergence
            if abs(previous_loss - loss_value) < convergence_threshold:
                print(f"Converged at iteration {iteration_count} (change < {convergence_threshold:.2e})")
            
            previous_loss = loss_value
            
            return loss
        
        # Run L-BFGS optimization
        while iteration_count < max_iterations:
            optimizer.step(closure)
            
            # Check for early stopping based on convergence
            if iteration_count > 10 and abs(self.loss_history[-1] - self.loss_history[-10]) < convergence_threshold:
                print(f"\nEarly stopping: Loss converged (change < {convergence_threshold:.2e})")
                break
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters in the network."""
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint.
        """
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'loss_history': self.loss_history,
            'iteration': self._iteration
        }, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to the checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.loss_history = checkpoint['loss_history']
        self._iteration = checkpoint['iteration']
        print(f"Checkpoint loaded from {filepath}")
