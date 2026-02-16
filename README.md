# Physics-Informed Neural Networks for Burgers' Equation

PyTorch implementation of PINNs for solving the 1D viscous Burgers' equation.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)

[Live Demo](https://physics-informed-neural-network.streamlit.app)

## Overview

Physics-Informed Neural Networks embed physical laws (PDEs) directly into the loss function. This implementation solves the 1D viscous Burgers' equation:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

with initial condition $u(x, 0) = -\sin(\pi x)$ and boundary conditions $u(\pm 1, t) = 0$.

## Features

- Modular architecture with clean separation of concerns
- Two-stage optimization: Adam + L-BFGS
- Automatic differentiation for computing PDE residuals
- Interactive Streamlit web app with 3D visualizations
- Full type hints and PEP 8 compliance

## Installation

```bash
git clone https://github.com/m-kuehnle/PINNs.git
cd PINNs
pip install -r requirements.txt
```

## Quick Start

**Interactive Web App:**

```bash
streamlit run streamlit_app.py
```

**Command Line:**

```bash
cd examples
python burgers_1d.py
```

## Project Structure

```
PINNs/
├── src/
│   ├── pinn/           # Network and trainer
│   ├── pde/            # Burgers equation solver
│   └── utils/          # Data generation and visualization
├── examples/           # Example scripts
├── streamlit_app.py    # Interactive web app
└── requirements.txt
```

## Network Architecture

- Input: (x, t) → Output: u(x, t)
- 3 hidden layers × 50 neurons
- Activation: tanh
- ~7,851 parameters

## Training

1. Adam optimizer: 10,000 iterations (fast convergence)
2. L-BFGS optimizer: up to 1,000 iterations (high precision)

Results saved to `results/` including 3D plots, comparisons, and trained model.

## License

MIT License
