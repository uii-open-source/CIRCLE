# Use the official PyTorch CUDA-enabled runtime image
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install git (not included in the minimal runtime image)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone the source code from GitHub
RUN git clone https://github.com/uii-open-source/CIRCLE.git CIRCLE

# Switch to the cloned project directory
WORKDIR /workspace/CIRCLE

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add the current code directory to the Python path
ENV PYTHONPATH="${PYTHONPATH}:/workspace/CIRCLE"
