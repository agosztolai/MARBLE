#!/bin/bash
# adapted from https://github.com/BlueBrain/morphoclass/blob/main/install.sh

echo "Checking the PyTorch version"
SED=$(which gsed || which sed)
TORCH_VERSION=$(pip freeze | grep torch== | $SED -re "s/torch==([^+]+).*/\1/")

if [ -z "$TORCH_VERSION" ]
then
  echo ">> PyTorch should have been installed, but no installation of it was found"
  echo ">> Installation failed"
  exit 1
fi

echo "Checking the CUDA version"
OS_NAME=$(uname -s)
if [ "$OS_NAME" == "Linux" ]
then
  # check if cuda is available in these 3 locations
  if [ ! -d "/usr/local/cuda/lib64" ] && [ ! -d "/usr/lib/cuda/lib64" ] && [ ! -d "/usr/lib/nvidia-cuda-toolkit" ]
  then
    echo ">> PyTorch was installed with CUDA support, but no CUDA toolkit libraries were found."
    echo ">> Re-installing the CPU version of PyTorch"
    echo ">> PyTorch version: $TORCH_VERSION"
    pip install "torch==${TORCH_VERSION}+cpu" -f https://download.pytorch.org/whl/torch_stable.html
  fi
fi


echo "Installing PyTorch-Geometric"
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda or '')")
if [ -z "$CUDA_VERSION" ]
then
  CUDA="cpu"
else
  CUDA="cu${CUDA_VERSION/./}"
fi

FIND_LINKS="https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA}.html"
echo "$FIND_LINKS"

# Indexes like devpi take precedence over find-links. But the currently
# configured index might have some versions of the given package too. If the
# index only has the source distribution then instead of downloading the wheel
# from the find-links URL pip will download the source distribution from
# the index and try to build a wheel form it. To avoid this and to force lookup
# in the find-links URL disable the index completely.
pip install torch-scatter #--no-index -f "$FIND_LINKS"
pip install torch-sparse #--no-index -f "$FIND_LINKS"
pip install torch-cluster #--no-index -f "$FIND_LINKS"
pip install torch-spline-conv #--no-index -f "$FIND_LINKS"
pip install "torch-geometric"
