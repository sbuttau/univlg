FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    sudo \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

ENV CUDA_HOME=/usr/local/cuda-11.8
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# ---- Install PyTorch + CUDA 11.8 ----
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# ---- torch-scatter for PyTorch 2.1 + cu118 ----
RUN pip install torch-scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
RUN python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
# ---- Detectron2 wheel for torch 2.1 + cu118 ----
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2 --no-build-isolation

# ---- Flash-attn (works with torch 2.1) ----
RUN pip install psutil numpy
RUN pip install flash-attn==2.6.3 --no-build-isolation

# Workspace
WORKDIR /workspace/univlg
COPY . /workspace/univlg

# Install project deps
RUN pip install -r /workspace/univlg/docs/requirements.txt

# NLP optional stuff
RUN pip install spacy
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('stopwords')"

RUN pip install git+https://github.com/waspinator/pycococreator.git

# Init script
RUN bash docs/init.sh

CMD ["/bin/bash"]
