#!/bin/bash

set -e
set -o pipefail

echo "🔧 Starting Falcon local setup..."

# Clone robustness locally into Falcon-adapted/
if [ ! -d robustness ]; then
  echo "📦 Cloning robustness..."
  git clone https://github.com/MadryLab/robustness robustness
fi
pip install -e ./robustness

# Clone GMatch4py locally into Falcon-adapted/
if [ ! -d GMatch4py ]; then
  echo "📦 Cloning GMatch4py..."
  git clone https://github.com/jacquesfize/GMatch4py GMatch4py
fi
pip install -e ./GMatch4py

# Install any additional requirements
echo "📦 Installing scikit-misc..."
pip install scikit-misc

echo "✅ Falcon setup complete (with local dependencies)."
