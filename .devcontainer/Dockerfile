FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    sudo \
    ssh \
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

    # ---- Detectron2 wheel for torch 2.1 + cu118 ----
RUN git clone https://github.com/facebookresearch/detectron2.git \
    && python -m pip install -e detectron2 --no-build-isolation

# ---- Flash-attn (works with torch 2.1) ----
RUN pip install psutil numpy
RUN pip install flash-attn==2.6.3 --no-build-isolation

# Workspace
WORKDIR /workspaces/univlg
COPY ./docs ./docs

# Install project deps
RUN pip install -r ./docs/requirements.txt

# NLP optional stuff
RUN pip install spacy
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('stopwords')"

RUN pip install git+https://github.com/waspinator/pycococreator.git
RUN pip install transformers==4.44
COPY ./libs ./libs
ENV TORCH_CUDA_ARCH_LIST="6.1"
WORKDIR /workspaces/univlg/libs/pointops2
RUN pip install --no-build-isolation -e . 

CMD ["/bin/bash"]

# Add users
ARG GROUP_NAME=vimp
ARG DEVOPS_GID=2000

ARG USERNAME=sara
ARG USER_UID=1001
ARG USER_GID=1001

# create the user with sudo rights
RUN groupadd --gid $DEVOPS_GID $GROUP_NAME \
    && useradd --uid $USER_UID --gid $DEVOPS_GID -s /bin/bash -m $USERNAME \
    && echo "%$GROUP_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
