#!/bin/bash
echo "🔬 Running Zeta Confinement Demo..."
cd "$(dirname "$0")"
python zeta_confinement_patched.py
echo "✅ Check generated plots in ../data/"
