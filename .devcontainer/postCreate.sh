#!/usr/bin/env bash
set -e

echo "[post-create] config Git..."
git config --global user.name "sbuttau"
git config --global user.email "sara.buttau@hotmail.it"

echo "[post-create] clone PyTorch3D..."
cd /workspaces/univlg
if [ ! -d "pytorch3d" ]; then
    git clone https://github.com/facebookresearch/pytorch3d.git
fi

echo "[post-create] install PyTorch3D..."
cd /workspaces/univlg/pytorch3d
sudo pip install --no-build-isolation -e .

echo "[post-create] Build pixel_decoder ops..."
cd /workspaces/univlg/univlg/modeling/pixel_decoder/ops
sudo rm -rf build
sudo bash make.sh

echo "[post-create] exec docs/init.sh..."
cd /workspaces/univlg
bash docs/init.sh

echo "[post-create] completed!"
