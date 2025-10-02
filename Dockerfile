# Stage 1: Build environment based on the required Python version
FROM python:3.10-slim AS core

# 1. Install essential system dependencies for building code, git, and file management
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Add the 'uv' installer for fast Python package management
# See https://github.com/astral-sh/uv
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /usr/local/bin/uv
# Configure uv to use the system-wide Python interpreter
ENV UV_SYSTEM_PYTHON=1

# 3. Set up the working directory and the essential SC2PATH environment variable
WORKDIR /app
ENV SC2PATH="/app/3rdparty/StarCraftII"

# 4. Copy all your project files into the Docker image
COPY . .

# 5. --- CRITICAL: Patch Source Files Before Installation ---
# This step applies all the fixes we discovered during debugging.
RUN set -e; \
    echo "Applying source code patches..."; \
    # Fix broken pyyaml version and add missing packages to requirements.txt
    sed -i 's/pyyaml==5.4.1/PyYAML>=6.0/' requirements.txt; \
    echo "sacred" >> requirements.txt; \
    echo "gymnasium" >> requirements.txt; \
    echo "pettingzoo" >> requirements.txt; \
    # Fix the yaml.load() error in the main script
    sed -i "s/config_dict = yaml.load(f)/config_dict = yaml.load(f, Loader=yaml.FullLoader)/" src/main.py

# 6. Install Python dependencies, handling the PyTorch version based on CUDA availability
# Build argument to control CUDA installation. Default is true.
ARG USE_CUDA=true
RUN set -e; \
    echo "Installing Python dependencies..."; \
    # Install cython first, as it's a build dependency for other packages
    uv pip install cython; \
    # Install all packages from the now-corrected requirements.txt
    uv pip install -r requirements.txt; \
    # If USE_CUDA is true, uninstall the default CPU-only PyTorch (if any)
    # and install the version compatible with CUDA 12.1
    if [ "$USE_CUDA" = true ] ; \
    then \
      echo "Installing PyTorch with CUDA 12.1 support..."; \
      uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    else \
      echo "USE_CUDA is false. Using CPU-only PyTorch."; \
    fi

# 7. Install StarCraft II
RUN bash install_sc2.sh

# 8. Clone and Manually Patch the SMAC Library
RUN set -e; \
    echo "Patching SMAC library..."; \
    git clone https://github.com/oxwhirl/smac.git; \
    uv pip install -e smac/; \
    # Manually add the missing get_map_params function to the library's __init__.py
    echo -e "\ndef get_map_params(map_name):\n    return map_param_registry[map_name]\n" >> smac/smac/env/starcraft2/maps/__init__.py; \
    # Manually create the map directory and copy the map files into the game folder
    mkdir -p "$SC2PATH/Maps/SMAC_Maps/"; \
    cp smac_patch/SMAC_Maps/* "$SC2PATH/Maps/SMAC_Maps/"

# 9. Expose the TensorBoard port so you can view logs from outside the container
EXPOSE 6006

# 10. Set the default command to start a bash session.
# This allows you to easily attach to the container and run experiments manually.
CMD ["bash"]
