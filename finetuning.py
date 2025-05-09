import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

# from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from processing_paligemma import PaliGemmaProcessor
from modeling_paligemma import PaliGemmaForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from data import ClevrerDataset
import torchvision.transforms as transforms
import numpy as np
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from transformers import Trainer
from transformers import TrainingArguments
from huggingface_hub import login
import os
import wandb

from datetime import datetime
import pytz

ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

# print("Current time in IST:", now_ist.strftime("%Y-%m-%d %H:%M:%S %Z%z"))


def setup_wandb(project_name, run_name):
    try:
        wandb.login(key="c05e9a6ff01ac9550c6c83b7c666c67f0d688723")
        print("Successfully logged into WandB.")
    except Exception as e:
        print(f"Error logging into WandB: {e}")
    
    # Optional: Log models
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "true"
    
    try:
        wandb.init(project=project_name, name=run_name)
        print(f"WandB run initialized: Project - {project_name}, Run - {run_name}")
    except Exception as e:
        print(f"Error initializing WandB run: {e}")

setup_wandb(project_name="Physical_Reasoning", run_name=now_ist.strftime("%m-%d %H:%M"))

login(token="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")

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

dataset = ClevrerDataset(
    frames_root='frame_captures',
    json_path='train.json',
    transform=transform,
    question_type="all"
)

train_ds, val_ds = dataset.train_test_split(test_size=0.2)

model_id = "google/paligemma2-3b-mix-448"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "fc1", "fc2", "linear", "patch_embedding", "position_embedding", "embed_tokens"],
    task_type="CAUSAL_LM",
)

# device = "cuda:2" if torch.cuda.is_available() else "cpu"
# device = "cuda"
# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto")#, quantization_config=bnb_config)
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

DTYPE = model.dtype
processor = PaliGemmaProcessor.from_pretrained(model_id)

def clevrer_collate_fn(examples):
    prompts, labels, images = [], [], []
    for example in examples:
        if example['question_type'] != 'descriptive':
            text = "Select all that apply. " + example["question"]
        else:
            text = example["question"]

        image_tokens = "<image> " * len(example['frames'])
        prompt = image_tokens + text + " en"
        prompts.append(prompt)
        labels.append(example['answer'])
        images.append(example['frames'])
    
    tokens = processor(
        text=prompts,
        images=images,
        return_tensors='pt',
        do_rescale=False,
        padding="longest",
        suffix=labels,
    ).to(DTYPE)
    # ).to(device, DTYPE)

    return tokens


args=TrainingArguments(
            num_train_epochs=3,
            remove_unused_columns=False,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=5e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            optim="adamw_torch", # you can use paged optimizers like paged_adamw_8bit for QLoRA
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=1,
            output_dir="./paligemma_clevrer",
            bf16=True,
            report_to="wandb",
            dataloader_pin_memory=False,
            label_names=["labels"]
        )

trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        data_collator=clevrer_collate_fn,
        args=args
        )

trainer.train()