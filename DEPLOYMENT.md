# Deploying the Interactive PINN App

This project includes an interactive Streamlit web application for visualizing Physics-Informed Neural Networks solving the 1D Burgers' Equation.

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Fork/Push to GitHub** (already done ✅)

   ```
   https://github.com/m-kuehnle/PINNs
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository: `m-kuehnle/PINNs`
   - Main file path: `streamlit_app.py`
   - Click "Deploy"

3. **App URL** (after deployment):
   ```
   https://m-kuehnle-pinns.streamlit.app
   ```

### Option 2: Hugging Face Spaces

1. **Create a new Space**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Select "Streamlit" as the SDK
   - Name: `pinns-burgers-equation`
   - Clone the space repository

2. **Copy files to the Space**

   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/pinns-burgers-equation
   cd pinns-burgers-equation

   # Copy files from this repo
   cp -r ../PINNs/src .
   cp ../PINNs/streamlit_app.py app.py
   cp ../PINNs/requirements.txt .

   git add .
   git commit -m "Add PINN Streamlit app"
   git push
   ```

3. **App URL** (after deployment):
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/pinns-burgers-equation
   ```

### Option 3: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 🎯 Features

- **Interactive 3D Visualization**: Rotate and zoom the solution surface
- **Parameter Control**: Adjust viscosity, network architecture, and training settings
- **Real-time Training**: Train PINNs directly in the browser
- **Loss Monitoring**: Track convergence with interactive plots
- **Responsive Design**: Works on desktop and mobile devices

## 📊 Performance Notes

**Training Time:**

- Default settings (~5000 Adam + 500 L-BFGS iterations): 2-5 minutes
- On CPU: slower, recommend reducing iterations
- On GPU (CUDA/MPS): faster, can use more iterations

**Streamlit Cloud Limitations:**

- CPU-only (no GPU)
- 1GB RAM limit
- Recommend using default or reduced training iterations

## 🔧 Configuration

The app caches trained models using `@st.cache_resource`, so:

- Same parameters = instant reload (no retraining)
- Different parameters = new training run

## 📝 Notes

- First deployment may take 5-10 minutes to build
- Streamlit Cloud auto-restarts on git push
- For production, consider pre-training and loading saved models

## 🚀 Quick Deploy Command

For Streamlit Cloud, just push to GitHub and follow the web interface.

For Hugging Face, use the included `README.md` in the space.
