---
title: PINN Burgers Equation Solver
emoji: 🌊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: streamlit_app.py
pinned: false
license: mit
---

# Physics-Informed Neural Network for Burgers' Equation

An interactive web application for solving the 1D viscous Burgers' equation using Physics-Informed Neural Networks (PINNs).

## Features

- 🎨 Interactive 3D visualization of the solution
- ⚙️ Configurable physics and network parameters
- 📊 Real-time training and loss monitoring
- 🧠 Two-stage optimization (Adam + L-BFGS)

## How to Use

1. Configure parameters in the sidebar
2. Click "Train PINN" to start training
3. Explore the interactive 3D solution
4. Adjust parameters and retrain to see different behaviors

## Mathematical Formulation

Solving the viscous Burgers' equation:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

with initial condition $u(x, 0) = -\sin(\pi x)$ and boundary conditions $u(\pm 1, t) = 0$.

## GitHub

[View Source Code](https://github.com/m-kuehnle/PINNs)
