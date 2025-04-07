#!/bin/bash

# Script to automate Git repository setup for Reservoir Computing project
# Usage: ./setup_git_repo.sh [github_username] [repo_name]

# Set default values
GITHUB_USERNAME=${1:-"yourusername"}
REPO_NAME=${2:-"reservoir-computing"}
REPO_DESCRIPTION="A modular Reservoir Computing framework for chaotic systems prediction and anomaly detection"

echo "🚀 Starting Git repository setup..."

# 1. Initialize Git repository
echo "✅ Initializing Git repository..."
git init

# 2. Create comprehensive .gitignore
echo "✅ Creating .gitignore file..."
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebooks
.ipynb_checkpoints

# Data files (optional - you might want to keep small ones)
data/*
!data/.gitkeep

# Results & logs (optional - you might want to keep some)
results/*
!results/.gitkeep
logs/*
!logs/.gitkeep

# Virtual environments
venv/
ENV/

# IDE specific files
.idea/
.vscode/
*.swp
EOL

# Make sure the .gitkeep files exist
mkdir -p data results logs
touch data/.gitkeep results/.gitkeep logs/.gitkeep

git add .gitignore
git commit -m "Add comprehensive .gitignore for Python ML project"

# 3. Add files in logical chunks

# Check if directories exist before adding them
if [ -d "src/ml" ]; then
    echo "✅ Adding core reservoir computing components..."
    git add src/ml/reservoir_builder.py src/ml/training_manager.py src/ml/autonomous_runner.py 2>/dev/null
    git commit -m "Add core reservoir computing components" || echo "No core components found, skipping"

    echo "✅ Adding evaluation and model persistence modules..."
    git add src/ml/evaluation_manager.py src/ml/model_persistence.py 2>/dev/null
    git commit -m "Add evaluation and model persistence modules" || echo "No evaluation modules found, skipping"
    
    echo "✅ Adding main applications..."
    git add src/ml/main_ml.py 2>/dev/null
    git commit -m "Add main script for chaotic systems prediction" || echo "Main script not found, skipping"
    
    git add src/ml/ecg_anomaly_detector.py src/ml/ecg_online_detector.py 2>/dev/null
    git commit -m "Add ECG anomaly detection implementations" || echo "ECG detection scripts not found, skipping"
    
    git add src/ml/config_example.yaml 2>/dev/null
    git commit -m "Add configuration example" || echo "Config example not found, skipping"
fi

if [ -d "src/simulators" ]; then
    echo "✅ Adding system simulators..."
    git add src/simulators/base.py src/simulators/lorenz.py src/simulators/ecg_simulator.py 2>/dev/null
    git commit -m "Add system simulators and ECG adapter" || echo "Simulators not found, skipping"
fi

# Documentation
echo "✅ Adding documentation..."
git add README.md 2>/dev/null
git commit -m "Add project README" || echo "README not found, skipping"

git add ecg_anomaly_detection_results.md 2>/dev/null
git commit -m "Add ECG analysis results" || echo "ECG results document not found, skipping"

# Requirements
echo "✅ Adding requirements file..."
git add requirements.txt 2>/dev/null
git commit -m "Add dependencies list" || echo "Requirements file not found, skipping"

# Add any remaining files
echo "✅ Adding any remaining files..."
git add .
git commit -m "Add remaining project files" || echo "No additional files to commit"

# 4. Check for GitHub CLI
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI found. Checking authentication..."
    
    # Check authentication status
    if gh auth status &> /dev/null; then
        echo "✅ Already authenticated with GitHub"
    else
        echo "⚠️ Please authenticate with GitHub CLI:"
        gh auth login
    fi
    
    # 5. Create GitHub repository
    echo "✅ Creating GitHub repository: $REPO_NAME..."
    gh repo create "$REPO_NAME" --public --description "$REPO_DESCRIPTION" --source=. --remote=origin --push
else
    echo "⚠️ GitHub CLI not found. Setting up remote manually..."
    
    # 6. Set remote origin
    echo "✅ Setting remote origin..."
    git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    
    # 7. Configure for larger files
    echo "✅ Configuring Git for larger files..."
    git config http.postBuffer 524288000
    
    # 8. Push to GitHub
    echo "✅ Pushing to GitHub..."
    git branch -M main
    git push -u origin main
    
    echo "⚠️ If the push failed, please create the repository manually at:"
    echo "    https://github.com/new"
    echo "Then push again with: git push -u origin main"
fi

# 9. Confirm success
echo "✅ Checking Git status..."
git status

echo "🎉 Git repository setup completed!"
echo ""
echo "Your repository should now be available at: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "Next steps you might want to consider:"
echo "1. Add a license file (MIT recommended)"
echo "2. Set up GitHub Actions for CI/CD"
echo "3. Create a GitHub Pages site to showcase your project"
echo ""
echo "Enjoy showcasing your Reservoir Computing project! 🚀" 