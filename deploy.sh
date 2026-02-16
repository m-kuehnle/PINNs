#!/bin/bash

# Quick deployment script for PINN Streamlit app
# Run this to push your changes and get deployment instructions

echo "=============================================="
echo "  PINN Streamlit App - Deployment Script"
echo "=============================================="
echo ""

# Check if in correct directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found!"
    echo "Please run this script from the PINNs directory"
    exit 1
fi

echo "✅ Found streamlit_app.py"
echo ""

# Git status
echo "📊 Current Git Status:"
echo "----------------------"
git status --short
echo ""

# Ask for commit
read -p "Do you want to commit and push changes? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "📝 Committing changes..."
    git add .
    git commit -m "Add interactive Streamlit app with 3D PINN visualizations"
    
    echo ""
    echo "🚀 Pushing to GitHub..."
    git push origin main
    
    echo ""
    echo "=============================================="
    echo "  ✅ SUCCESS! Your code is on GitHub"
    echo "=============================================="
    echo ""
    echo "🌐 NEXT STEPS:"
    echo ""
    echo "1. Go to: https://share.streamlit.io"
    echo "2. Sign in with GitHub"
    echo "3. Click 'New app'"
    echo "4. Enter:"
    echo "   - Repository: m-kuehnle/PINNs"
    echo "   - Branch: main"
    echo "   - Main file: streamlit_app.py"
    echo "5. Click 'Deploy!'"
    echo ""
    echo "Your app will be live at:"
    echo "👉 https://m-kuehnle-pinns.streamlit.app"
    echo ""
    echo "=============================================="
else
    echo ""
    echo "ℹ️  Skipped deployment. Run this script again when ready!"
fi

echo ""
echo "📖 For detailed instructions, see:"
echo "   - DEPLOY_NOW.md"
echo "   - DEPLOYMENT.md"
echo ""
