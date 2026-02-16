# Quick Deployment Guide

## Your PINN App is Ready to Deploy

### What You Have Now:

**Interactive Streamlit App** (`streamlit_app.py`)

- Beautiful 3D visualizations with Plotly
- Real-time parameter tuning
- Live PINN training in the browser

**Deployment Configurations**

- `.streamlit/config.toml` - App theming and settings
- `requirements.txt` - Updated with streamlit and plotly
- `README_HF.md` - For Hugging Face Spaces

---

## Deploy to Streamlit Cloud (Easiest)

### Step 1: Push Latest Changes to GitHub

```bash
cd /Users/manuel/Coding/PINNs

# Add all new files
git add .
git commit -m "Add interactive Streamlit app with 3D visualizations"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. **Go to:** [share.streamlit.io](https://share.streamlit.io)

2. **Sign in** with your GitHub account

3. **Click** "New app" (or "Create app")

4. **Fill in:**
   - Repository: `m-kuehnle/PINNs`
   - Branch: `main`
   - Main file path: `streamlit_app.py`

5. **Click** "Deploy!"

6. **Wait** 2-5 minutes for build and deployment

7. **Your app will be live at:**
   ```
   https://m-kuehnle-pinns.streamlit.app
   ```

**That's it!** Your interactive PINN app is now live!

---

## Alternative: Deploy to Hugging Face Spaces

### Step 1: Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Space name: `pinns-burgers-equation`
3. License: `MIT`
4. Select SDK: **Streamlit**
5. Click "Create Space"

### Step 2: Upload Files

You can either:

**Option A: Use the Web Interface**

- Click "Files" → "Add file" → "Upload files"
- Upload: `streamlit_app.py` (rename to `app.py`), `requirements.txt`, and the entire `src/` folder

**Option B: Use Git**

```bash
# Clone the space
git clone https://huggingface.co/spaces/YOUR_USERNAME/pinns-burgers-equation
cd pinns-burgers-equation

# Copy necessary files
cp -r /Users/manuel/Coding/PINNs/src .
cp /Users/manuel/Coding/PINNs/streamlit_app.py app.py
cp /Users/manuel/Coding/PINNs/requirements.txt .
cp /Users/manuel/Coding/PINNs/README_HF.md README.md

# Push to deploy
git add .
git commit -m "Deploy PINN Streamlit app"
git push
```

### Step 3: Access Your App

Your app will be available at:

```
https://huggingface.co/spaces/YOUR_USERNAME/pinns-burgers-equation
```

---

## Test Locally First

Before deploying, test the app on your machine:

```bash
cd /Users/manuel/Coding/PINNs

# Install dependencies (if not already)
pip install streamlit plotly

# Run the app
streamlit run streamlit_app.py
```

Open your browser at: `http://localhost:8501`

---

## What Users Will See

When visitors access your deployed app, they can:

1. **Configure Parameters** (sidebar):
   - Viscosity (ν)
   - Network architecture (layers, neurons)
   - Training iterations (Adam + L-BFGS)
   - Learning rate

2. **Train PINN** with one click

3. **Explore Interactive Visualizations**:
   - 🌊 **3D Surface Plot** - Rotate, zoom, pan the solution
   - **Initial vs Final** - Side-by-side comparison
   - **Loss Evolution** - Training convergence plot

4. **View Statistics**:
   - Final loss, minimum loss
   - Total iterations
   - Network parameters

---

## Performance Tips

### For Streamlit Cloud (Free Tier):

- **CPU only** (no GPU)
- **1 GB RAM limit**
- **Recommended settings**:
  - Adam iterations: 3000-5000
  - L-BFGS iterations: 300-500
  - Neurons per layer: 30-50

### For Hugging Face Spaces (Free Tier):

- **2 vCPUs, 16 GB RAM**
- **Better for longer training**
- **Recommended settings**:
  - Adam iterations: 5000-10000
  - L-BFGS iterations: 500-1000
  - Neurons per layer: 50-100

---

## Customization

### Change App Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"  # Your color
backgroundColor = "#0E1117"  # Dark theme
```

### Add Features

Edit `streamlit_app.py` to add:

- Different PDEs
- Pre-trained models
- Download trained models
- Export animations

---

## Monitoring

### Streamlit Cloud:

- Dashboard: [share.streamlit.io](https://share.streamlit.io)
- View logs, usage, and errors
- Auto-redeploys on git push

### Hugging Face Spaces:

- Dashboard: Your space page
- View build logs
- Restart/rebuild as needed

---

## Next Steps

1. **Push to GitHub** (if not already done)
2. **Deploy to Streamlit Cloud** (5 minutes)
3. **Share your app link** with colleagues!
4. **Update README.md** with your live app link

---

## Troubleshooting

**App won't start?**

- Check build logs for errors
- Verify `requirements.txt` is complete
- Ensure Python 3.8+ is specified

**Out of memory?**

- Reduce training iterations
- Decrease network size
- Add `@st.cache_resource` to heavy functions

**Slow training?**

- Expected on CPU (2-5 min for default settings)
- Consider pre-training and loading models
- Reduce Adam iterations for faster demos

---

## Support

- Streamlit Docs: [docs.streamlit.io](https://docs.streamlit.io)
- Hugging Face Docs: [huggingface.co/docs/hub](https://huggingface.co/docs/hub/spaces)
- Community: [discuss.streamlit.io](https://discuss.streamlit.io)

---

**Ready to deploy? Let's go! 🚀**
