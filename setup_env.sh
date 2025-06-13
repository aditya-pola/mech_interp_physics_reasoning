# Create new virtual environment with uv
uv venv .venv_new

# Activate the virtual environment
# On Windows: .venv_new\Scripts\activate
# On Unix: source .venv_new/bin/activate

# Install required packages
uv pip install torch torchvision transformers peft accelerate bitsandbytes
uv pip install Pillow numpy pyyaml
uv pip install wandb pytz

# If you need CUDA support, install the appropriate torch version:
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
