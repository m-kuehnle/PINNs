"""
Neural Network Architecture for Physics-Informed Neural Networks.

This module provides the Multi-Layer Perceptron (MLP) architecture used
for approximating solutions to partial differential equations.
"""

from typing import List
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with tanh activation functions.
    
    This network architecture is specifically designed for PINNs, using
    tanh activations which provide smooth, infinitely differentiable
    functions suitable for computing PDE residuals via automatic differentiation.
    
    Attributes:
        layers (nn.Sequential): Sequential container of linear layers and activations.
        
    Args:
        layer_sizes: List of integers specifying the number of neurons in each layer.
                    First element is input dimension, last is output dimension.
                    Example: [2, 50, 50, 50, 1] creates a network with 2 inputs,
                    3 hidden layers of 50 neurons each, and 1 output.
    
    Example:
        >>> network = MLP([2, 50, 50, 50, 1])
        >>> x = torch.randn(100, 2)
        >>> output = network(x)
        >>> output.shape
        torch.Size([100, 1])
    """
    
    def __init__(self, layer_sizes: List[int]) -> None:
        """
        Initialize the MLP with specified layer sizes.
        
        Args:
            layer_sizes: List of integers for layer dimensions.
        
        Raises:
            ValueError: If layer_sizes has fewer than 2 elements.
        """
        super(MLP, self).__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output dimensions")
        
        # Build network layers
        layers: List[nn.Module] = []
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add tanh activation for all layers except the output layer
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights using Xavier/Glorot initialization
        # This is particularly effective for tanh activations
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize network weights using Xavier uniform initialization.
        
        Xavier initialization helps maintain gradient magnitudes across layers,
        which is crucial for training deep networks with tanh activations.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.layers(x)
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters in the network.
        
        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
