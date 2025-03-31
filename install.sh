conda create -n physic python=3.11
conda activate physic
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers peft wandb pillow