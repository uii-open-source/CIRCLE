# Use the official PyTorch CUDA-enabled runtime image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install git (not included in the minimal runtime image)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

RUN pip install flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Clone the source code from GitHub
RUN git clone https://github.com/uii-open-source/CIRCLE.git CIRCLE

# Switch to the cloned project directory
WORKDIR /workspace/CIRCLE

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add the current code directory to the Python path
ENV PYTHONPATH="${PYTHONPATH}:/workspace/CIRCLE"
