"""
Interactive Physics-Informed Neural Network for 1D Burgers' Equation
Streamlit Web Application

This app allows users to train and visualize PINN solutions interactively.
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pinn.network import MLP
from src.pinn.trainer import PINNTrainer
from src.pde.burgers import BurgersPINN
from src.utils.data import generate_training_data, generate_test_grid, initial_condition


# Page configuration
st.set_page_config(
    page_title="PINN: Burgers' Equation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_or_train_model(nu, hidden_layers, neurons_per_layer, adam_iter, lbfgs_iter, adam_lr):
    """Train the PINN model with caching."""
    
    # Network setup
    layer_sizes = [2] + [neurons_per_layer] * hidden_layers + [1]
    network = MLP(layer_sizes=layer_sizes)
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # PINN setup
    pinn = BurgersPINN(network=network, nu=nu, device=device)
    
    # Generate training data
    data = generate_training_data(
        n_pde=10000,
        n_ic=100,
        n_bc=100,
        x_min=-1.0,
        x_max=1.0,
        t_min=0.0,
        t_max=1.0,
        device=device
    )
    
    # Define loss function
    def loss_fn():
        total_loss, _ = pinn.compute_loss(
            x_pde=data['x_pde'],
            t_pde=data['t_pde'],
            x_ic=data['x_ic'],
            t_ic=data['t_ic'],
            u_ic=data['u_ic'],
            x_bc=data['x_bc'],
            t_bc=data['t_bc']
        )
        return total_loss
    
    # Training
    trainer = PINNTrainer(network=network, device=device)
    history = trainer.train(
        loss_fn=loss_fn,
        adam_iterations=adam_iter,
        lbfgs_iterations=lbfgs_iter,
        adam_lr=adam_lr,
        print_every=max(adam_iter // 10, 1)
    )
    
    return pinn, history, device


def create_3d_surface_plot(X, T, U):
    """Create interactive 3D surface plot using Plotly."""
    fig = go.Figure(data=[go.Surface(
        x=X, y=T, z=U,
        colorscale='RdBu_r',
        colorbar=dict(title="u(x, t)"),
        hovertemplate='x: %{x:.3f}<br>t: %{y:.3f}<br>u: %{z:.3f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text="PINN Solution: 3D Surface Plot",
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#1f77b4')
        ),
        scene=dict(
            xaxis=dict(title='Space (x)', backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(title='Time (t)', backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(title='u(x, t)', backgroundcolor="rgb(230, 230,230)"),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def create_comparison_plot(x, u_initial, u_final):
    """Create initial vs final state comparison."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Initial Condition (t=0)', 'Final State (t=1)'),
        horizontal_spacing=0.12
    )
    
    # Initial condition
    fig.add_trace(
        go.Scatter(x=x, y=u_initial, mode='lines', name='u(x, 0)',
                   line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Final state
    fig.add_trace(
        go.Scatter(x=x, y=u_final, mode='lines', name='u(x, 1)',
                   line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Space (x)", row=1, col=1)
    fig.update_xaxes(title_text="Space (x)", row=1, col=2)
    fig.update_yaxes(title_text="u(x, t)", row=1, col=1)
    fig.update_yaxes(title_text="u(x, t)", row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Initial vs Final State Comparison",
        title_x=0.5,
        title_font=dict(size=18, color='#1f77b4')
    )
    
    return fig


def create_loss_plot(loss_history, adam_iter):
    """Create loss evolution plot."""
    iterations = np.arange(len(loss_history))
    
    fig = go.Figure()
    
    # Loss curve
    fig.add_trace(go.Scatter(
        x=iterations,
        y=loss_history,
        mode='lines',
        name='Total Loss',
        line=dict(color='blue', width=2)
    ))
    
    # Mark Adam/L-BFGS transition
    if adam_iter < len(loss_history):
        fig.add_vline(
            x=adam_iter,
            line_dash="dash",
            line_color="red",
            annotation_text="Adam → L-BFGS",
            annotation_position="top"
        )
    
    fig.update_layout(
        title="Loss Evolution During Training",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        yaxis_type="log",
        height=400,
        hovermode='x unified',
        title_font=dict(size=18, color='#1f77b4')
    )
    
    return fig


def main():
    # Header
    st.markdown('<div class="main-header">Physics-Informed Neural Network</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Solution of the 1D Burgers\' Equation</div>', unsafe_allow_html=True)
    
    # Mathematical formulation in expander
    with st.expander("Mathematical Formulation", expanded=False):
        st.latex(r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}")
        st.markdown("""
        **Domain:** $x \in [-1, 1]$, $t \in [0, 1]$
        
        **Initial Condition:** $u(x, 0) = -\sin(\pi x)$
        
        **Boundary Conditions:** $u(-1, t) = u(1, t) = 0$
        
        The PINN learns to satisfy the PDE, initial conditions, and boundary conditions simultaneously.
        """)
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    st.sidebar.subheader("Physics Parameters")
    nu = st.sidebar.slider(
        "Viscosity (ν)",
        min_value=0.001,
        max_value=0.1,
        value=0.01/np.pi,
        step=0.001,
        format="%.4f",
        help="Kinematic viscosity coefficient"
    )
    
    st.sidebar.subheader("Network Architecture")
    hidden_layers = st.sidebar.slider(
        "Hidden Layers",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of hidden layers"
    )
    
    neurons_per_layer = st.sidebar.slider(
        "Neurons per Layer",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="Number of neurons in each hidden layer"
    )
    
    st.sidebar.subheader("Training Parameters")
    adam_iter = st.sidebar.number_input(
        "Adam Iterations",
        min_value=1000,
        max_value=50000,
        value=5000,
        step=1000,
        help="Number of Adam optimization steps"
    )
    
    lbfgs_iter = st.sidebar.number_input(
        "L-BFGS Iterations",
        min_value=100,
        max_value=5000,
        value=500,
        step=100,
        help="Maximum L-BFGS optimization steps"
    )
    
    adam_lr = st.sidebar.select_slider(
        "Adam Learning Rate",
        options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        value=1e-3,
        format_func=lambda x: f"{x:.0e}"
    )
    
    # Train button
    if st.sidebar.button("Train PINN", type="primary"):
        st.session_state.train_requested = True
    
    # Training and visualization
    if st.session_state.get('train_requested', False):
        
        # Progress indicator
        with st.spinner('Training PINN... This may take a few minutes.'):
            try:
                # Train model
                pinn, history, device = load_or_train_model(
                    nu, hidden_layers, neurons_per_layer,
                    adam_iter, lbfgs_iter, adam_lr
                )
                
                # Success message
                st.success(f"Training complete! Final Loss: {history['total_loss'][-1]:.2e} | Device: {device}")
                
                # Generate test grid
                X, T = generate_test_grid(n_x=200, n_t=200)
                U = pinn.evaluate_solution(X, T)
                
                # Main visualization
                st.markdown("---")
                st.markdown("### Interactive 3D Visualization")
                
                # 3D Surface Plot
                fig_3d = create_3d_surface_plot(X, T, U)
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Two column layout for comparison and loss
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Initial vs Final State")
                    x_1d = X[:, 0]
                    u_initial = initial_condition(x_1d)
                    u_final = U[:, -1]
                    fig_comp = create_comparison_plot(x_1d, u_initial, u_final)
                    st.plotly_chart(fig_comp, use_container_width=True)
                
                with col2:
                    st.markdown("### Loss Evolution")
                    fig_loss = create_loss_plot(history['total_loss'], adam_iter)
                    st.plotly_chart(fig_loss, use_container_width=True)
                
                # Statistics
                st.markdown("---")
                st.markdown("### Training Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Final Loss", f"{history['total_loss'][-1]:.2e}")
                
                with col2:
                    st.metric("Min Loss", f"{min(history['total_loss']):.2e}")
                
                with col3:
                    st.metric("Total Iterations", len(history['total_loss']))
                
                with col4:
                    network_params = sum(p.numel() for p in pinn.network.parameters())
                    st.metric("Network Parameters", f"{network_params:,}")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.exception(e)
    
    else:
        # Initial state
        st.info("Configure parameters in the sidebar and click **Train PINN** to start.")
        
        # Show example visualization
        st.markdown("### What You'll Get")
        st.markdown("""
        - **Interactive 3D Surface Plot**: Rotate, zoom, and explore the solution in real-time
        - **Initial vs Final Comparison**: See how the solution evolves from the initial sine wave
        - **Loss Evolution**: Monitor the training convergence with both Adam and L-BFGS phases
        - **Real-time Statistics**: Track loss, iterations, and model complexity
        """)
        
        # Add example image or info
        st.markdown("---")
        st.markdown("""
        ### How PINNs Work
        
        Physics-Informed Neural Networks embed the physics (PDE) directly into the loss function:
        
        ```
        Loss = w₁·Loss_PDE + w₂·Loss_IC + w₃·Loss_BC
        ```
        
        The network learns to satisfy:
        1. **PDE Residual**: The Burgers' equation at collocation points
        2. **Initial Condition**: u(x,0) = -sin(πx)
        3. **Boundary Conditions**: u(±1,t) = 0
        
        Using automatic differentiation, derivatives are computed exactly through the network!
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            Built using PyTorch, Streamlit & Plotly | 
            <a href='https://github.com/m-kuehnle/PINNs' target='_blank'>View on GitHub</a>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
