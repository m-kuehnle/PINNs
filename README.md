# Physics-Informed Neural Networks for Burgers' Equation

<div align="center">

**A clean, modular PyTorch implementation of Physics-Informed Neural Networks (PINNs) for solving the 1D viscous Burgers' equation**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[🚀 Live Demo](https://m-kuehnle-pinns.streamlit.app) | [📖 Documentation](DEPLOYMENT.md) | [🤗 Hugging Face](https://huggingface.co/spaces)

</div>

---

## 🌐 Interactive Web App

**Try it live!** Train and visualize PINNs directly in your browser:

👉 **[Launch Interactive Demo](https://m-kuehnle-pinns.streamlit.app)**

Features:

- 🎨 **Interactive 3D visualization** with Plotly
- ⚙️ **Real-time parameter tuning** (viscosity, network architecture, training)
- 📊 **Live training** and loss monitoring
- 🔄 **Instant retraining** with different configurations

---

## 📋 Table of Contents

- [Overview](#overview)
- [Interactive Web App](#interactive-web-app)
- [Mathematical Background](#mathematical-background)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web App Deployment](#web-app-deployment)
- [Results](#results)
- [Implementation Details](#implementation-details)
- [Contributing](#contributing)
- [References](#references)

---

## 🎯 Overview

This repository provides a **production-ready implementation** of Physics-Informed Neural Networks (PINNs) to solve the **1D viscous Burgers' equation**. The implementation follows best practices in scientific machine learning with clean, modular code that adheres to PEP 8 standards.

### What are PINNs?

Physics-Informed Neural Networks are a class of deep learning methods that embed physical laws (PDEs) directly into the loss function. Unlike traditional numerical methods, PINNs:

- Learn continuous solutions rather than discrete approximations
- Naturally handle inverse problems
- Can incorporate sparse or noisy data
- Provide mesh-free solutions

---

## 📐 Mathematical Background

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

## ✨ Features

- ✅ **Modular Architecture:** Clean separation of concerns (network, trainer, PDE solver, utilities)
- ✅ **Type Hints:** Full type annotations for better code quality and IDE support
- ✅ **Two-Stage Optimization:** Adam (fast convergence) + L-BFGS (high precision)
- ✅ **Automatic Differentiation:** Physics residuals computed via `torch.autograd`
- ✅ **Comprehensive Visualization:** 3D surface plots, comparisons, loss evolution
- ✅ **PEP 8 Compliant:** Professional code standards with detailed docstrings
- ✅ **Reproducible:** Seed control for consistent results
- ✅ **Well-Documented:** Extensive comments and mathematical explanations

---

## 📁 Project Structure

```
PINNs/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── src/                       # Source code
│   ├── __init__.py
│   ├── pinn/                  # Core PINN components
│   │   ├── __init__.py
│   │   ├── network.py         # MLP with tanh activation
│   │   └── trainer.py         # Two-stage optimizer (Adam + L-BFGS)
│   ├── pde/                   # PDE-specific implementations
│   │   ├── __init__.py
│   │   └── burgers.py         # Burgers' equation PINN solver
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── data.py            # Data generation (IC, BC, collocation)
│       └── visualization.py   # Plotting functions
├── examples/                  # Example scripts
│   └── burgers_1d.py          # Main executable
└── results/                   # Output directory (auto-generated)
    ├── solution_3d.png
    ├── comparison.png
    ├── loss_history.png
    └── burgers_pinn_model.pt
```

---

## 🚀 Installation

### Prerequisites

- Python ≥ 3.8
- pip or conda

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/PINNs.git
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

**Dependencies:**

- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.24.0` - Numerical computations
- `scipy>=1.10.0` - Scientific computing
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization
- `streamlit>=1.28.0` - Interactive web app
- `plotly>=5.17.0` - Interactive 3D plots

---

## 🎮 Quick Start

### Option 1: Interactive Web App (Recommended)

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

---

## 🌐 Web App Deployment

Deploy your own interactive PINN app:

### Streamlit Cloud (1-Click Deploy)

1. Fork this repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your forked repo
4. Set main file: `streamlit_app.py`
5. Click "Deploy"! 🚀

Your app will be live at: `https://[your-username]-pinns.streamlit.app`

### Hugging Face Spaces

```bash
# Create a new Streamlit Space on Hugging Face
# Clone and copy files
git clone https://huggingface.co/spaces/YOUR_USERNAME/pinns-app
cd pinns-app
cp -r ../PINNs/src .
cp ../PINNs/streamlit_app.py app.py
cp ../PINNs/requirements.txt .
cp ../PINNs/README_HF.md README.md

# Push to deploy
git add .
git commit -m "Deploy PINN app"
git push
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

---

## 🎮 Quick Start (CLI)

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

### Training Progress Example

```
================================================================================
Stage 1: Adam Optimization
--------------------------------------------------------------------------------
Iteration      1/  10000 | Loss: 2.456789e+00
Iteration   1000/  10000 | Loss: 1.234567e-01
...
Iteration  10000/  10000 | Loss: 3.456789e-03

================================================================================
Stage 2: L-BFGS Optimization (Fine-tuning)
--------------------------------------------------------------------------------
Iteration      1/   1000 | Loss: 3.421234e-03
Iteration    100/   1000 | Loss: 1.234567e-05
...
Final loss: 8.765432e-06
```

---

## 📊 Results

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

## 🔧 Implementation Details

### Network Architecture

**Multi-Layer Perceptron (MLP):**

- **Input:** $(x, t) \in \mathbb{R}^2$
- **Hidden Layers:** 3 layers × 50 neurons
- **Activation:** $\tanh$ (smooth, infinitely differentiable)
- **Output:** $u(x, t) \in \mathbb{R}$
- **Initialization:** Xavier/Glorot uniform
- **Parameters:** ~7,851 trainable parameters

### Two-Stage Optimization

**Stage 1: Adam Optimizer**

- **Purpose:** Rapid initial convergence
- **Iterations:** 10,000
- **Learning Rate:** 0.001
- **Advantage:** Fast, stable for early training

**Stage 2: L-BFGS Optimizer**

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Guidelines

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Write comprehensive docstrings (Google style)
4. Add unit tests for new features
5. Update README for significant changes

---

## 📚 References

### Key Papers

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). _Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations_. Journal of Computational Physics, 378, 686-707.

2. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2017). _Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations_. arXiv:1711.10561.

### Additional Resources

- [Original PINN Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Automatic Differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Senior Research Engineer for Scientific Machine Learning**

_Specializing in Physics-Informed Neural Networks, Deep Learning for PDEs, and Scientific Computing_

---

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Scientific machine learning community
- Original PINN authors (Raissi, Perdikaris, Karniadakis)

---

<div align="center">

**If you find this repository useful, please consider giving it a ⭐!**

</div>
