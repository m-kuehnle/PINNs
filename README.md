# Physics-Informed Neural Networks for Burgers' Equation

<div align="center">

**PyTorch implementation of Physics-Informed Neural Networks (PINNs) for solving the 1D viscous Burgers' equation**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Live Demo](https://physics-informed-neural-network.streamlit.app) | [Documentation](DEPLOYMENT.md)

</div>

---

## Interactive Web App

**Try it live:** Train and visualize PINNs directly in your browser.

**[Launch Interactive Demo](https://physics-informed-neural-network.streamlit.app)**

Features:

- Interactive 3D visualization with Plotly
- Real-time parameter tuning (viscosity, network architecture, training)
- Live training and loss monitoring
- Instant retraining with different configurations

## Overview

This repository provides a **production-ready implementation** of Physics-Informed Neural Networks (PINNs) to solve the **1D viscous Burgers' equation**. The implementation follows best practices in scientific machine learning with clean, modular code that adheres to PEP 8 standards.

### What are PINNs?

Physics-Informed Neural Networks are a class of deep learning methods that embed physical laws (PDEs) directly into the loss function. Unlike traditional numerical methods, PINNs:

- Learn continuous solutions rather than discrete approximations
- Naturally handle inverse problems
- Can incorporate sparse or noisy data
- Provide mesh-free solutions

---

## Mathematical Background

### The 1D Burgers' Equation

We solve the viscous Burgers' equation:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

**Domain:** $x \in [-1, 1]$, $t \in [0, 1]$

**Initial Condition:**
$$u(x, 0) = -\sin(\pi x)$$

**Boundary Conditions:**
$$u(-1, t) = u(1, t) = 0$$

**Parameters:**

- $\nu = \frac{0.01}{\pi}$ (kinematic viscosity)

### PINN Loss Function

The total loss combines three components:

$$\mathcal{L}_{\text{total}} = w_{\text{PDE}} \mathcal{L}_{\text{PDE}} + w_{\text{IC}} \mathcal{L}_{\text{IC}} + w_{\text{BC}} \mathcal{L}_{\text{BC}}$$

Where:

1. **PDE Residual Loss:**
   $$\mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{PDE}}} \sum_{i=1}^{N_{\text{PDE}}} \left| \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} \right|^2$$

2. **Initial Condition Loss:**
   $$\mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{IC}}} \sum_{i=1}^{N_{\text{IC}}} |u(x_i, 0) - u_0(x_i)|^2$$

3. **Boundary Condition Loss:**
   $$\mathcal{L}_{\text{BC}} = \frac{1}{N_{\text{BC}}} \sum_{i=1}^{N_{\text{BC}}} |u(x_{\text{boundary}}, t_i)|^2$$

Derivatives are computed via **automatic differentiation** using PyTorch's autograd.

---

## Installation

### Prerequisites

- Python ≥ 3.8
- pip or conda

### Step 1: Clone the Repository

```bash
git clone https://github.com/m-kuehnle/PINNs.git
cd PINNs
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n pinn python=3.10
conda activate pinn
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Interactive Web App

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501` and interact with the PINN in real-time!

### Option 2: Command Line Script

```bash
cd examples
python burgers_1d.py
```

### Expected Output

The script will:

1. **Initialize** the neural network (3 hidden layers, 50 neurons each)
2. **Generate** training data (10,000 PDE points, 100 IC points, 200 BC points)
3. **Train** using Adam (10,000 iterations) + L-BFGS (up to 1,000 iterations)
4. **Evaluate** the solution on a 200×200 grid
5. **Visualize** results (3D plot, comparison, loss evolution)
6. **Save** model and figures to `results/`

## Results

After training, you'll find in the `results/` directory:

### 1. 3D Surface Plot (`solution_3d.png`)

High-quality visualization of $u(x, t)$ showing:

- Shock wave formation
- Diffusion effects
- Boundary condition enforcement

### 2. Initial vs Final Comparison (`comparison.png`)

Side-by-side plots showing:

- Initial sine wave: $u(x, 0) = -\sin(\pi x)$
- Final diffused state: $u(x, 1)$

### 3. Loss Evolution (`loss_history.png`)

Semi-log plot displaying:

- Loss convergence during Adam phase
- Fine-tuning during L-BFGS phase
- Transition marker between optimization stages

### 4. Trained Model (`burgers_pinn_model.pt`)

Checkpoint containing:

- Model state dictionary
- Loss history
- Training iteration count

---

## Implementation Details

### Network Architecture

**Multi-Layer Perceptron (MLP):**

- **Input:** $(x, t) \in \mathbb{R}^2$
- **Hidden Layers:** 3 layers × 50 neurons
- **Activation:** $\tanh$ (smooth, infinitely differentiable)
- **Output:** $u(x, t) \in \mathbb{R}$
- **Initialization:** Xavier/Glorot uniform
- **Parameters:** ~7,851 trainable parameters

### Two-Stage Optimization

#### Stage 1: Adam Optimizer

- **Purpose:** Rapid initial convergence
- **Iterations:** 10,000
- **Learning Rate:** 0.001
- **Advantage:** Fast, stable for early training

#### Stage 2: L-BFGS Optimizer

- **Purpose:** High-precision fine-tuning
- **Max Iterations:** 1,000
- **Line Search:** Strong Wolfe conditions
- **Advantage:** Quasi-Newton method with superlinear convergence

### Automatic Differentiation

Derivatives computed using PyTorch's autograd:

```python
# First derivative: ∂u/∂t
u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

# Second derivative: ∂²u/∂x²
u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
```

### Data Sampling Strategy

- **PDE Collocation:** 10,000 random points in $(x, t) \in [-1, 1] \times [0, 1]$
- **Initial Condition:** 100 uniform points along $x$ at $t = 0$
- **Boundary Condition:** 100 random points at each boundary ($x = \pm 1$)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
