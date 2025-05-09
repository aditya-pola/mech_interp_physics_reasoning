conda create -n physic python=3.11
conda activate physic
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft wandb pillow pytz bitsandbytes