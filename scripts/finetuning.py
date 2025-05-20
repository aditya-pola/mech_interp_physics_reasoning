import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

import torch
import numpy as np
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from transformers import Trainer, TrainingArguments
from huggingface_hub import login as hf_login
import yaml
import sys
from datetime import datetime
import pytz

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

HOME_DIR = config.get('HOME')
sys.path.insert(0, HOME_DIR)

from src.processing_paligemma import PaliGemmaProcessor
from src.modeling_paligemma import PaliGemmaForConditionalGeneration
from src.utils import lora_filter, freeze_vision_tower, setup_wandb, load_dataset, generate_run_name, create_valid_dirname

ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

# print("Current time in IST:", now_ist.strftime("%Y-%m-%d %H:%M:%S %Z%z"))

model_config = config.get('model_train', {})
lora_config_params = config.get('lora', {})
data_config = config.get('data_config', {})

run_name = generate_run_name(config)
print(f"Run ---> {run_name}")

setup_wandb(project_name="Physical_Reasoning", run_name=run_name, key="c05e9a6ff01ac9550c6c83b7c666c67f0d688723")
hf_login(token="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")

data_config['HOME'] = HOME_DIR
train_ds, val_ds = load_dataset(data_config)

# model_id = "google/paligemma2-3b-mix-224"
model_id = model_config.get('model')
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, 
                                                          device_map="auto",
                                                          attn_implementation="eager",
                                                          token_compression=model_config.get('token_compression'),
                                                          target_length=model_config.get('target_length'))#, quantization_config=bnb_config)

# from transformers import AutoProcessor, AutoModelForImageTextToText
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B")
# model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B", device_map="auto")

target_modules = lora_filter(
        model=model,
        layers=lora_config_params.get('layers', "all"),
        layer_types=lora_config_params.get('layer_types', None),
        include_vision=lora_config_params.get('include_vision', True),
        include_language=lora_config_params.get('include_language', True),
        vision_layers=lora_config_params.get('vision_layers', None),
        language_layers=lora_config_params.get('language_layers', None)
    )

lora_config = LoraConfig(
    r=lora_config_params.get('rank'),
    target_modules=target_modules,
    task_type="CAUSAL_LM",
)

model.enable_input_require_grads()
model = get_peft_model(model, lora_config)

# freeze_vision_tower(model)

model.print_trainable_parameters()

DTYPE = model.dtype
processor = PaliGemmaProcessor.from_pretrained(model_id)

def clevrer_collate_fn(examples):
    prompts, labels, images = [], [], []
    for example in examples:
        if example['question_type'] != 'descriptive':
            text = "Select all that apply. " + example["question"]
        else:
            text = "Answer with one word or number only. " + example["question"]

        num_image_tokens = model_config['target_length'] * data_config['num_frames'] / 1024

        # image_tokens = "<image> " * len(example['frames'])
        image_tokens = "<image> " * int(num_image_tokens)
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

valid_dirname = create_valid_dirname(run_name)
output_dir = os.path.join(HOME_DIR, model_config.get('save_dir'), valid_dirname)

args=TrainingArguments(
            num_train_epochs=model_config.get('num_epochs', 3),
            remove_unused_columns=False,
            per_device_train_batch_size=model_config.get('batch_size'),
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=float(model_config.get('learning_rate')),
            weight_decay=float(model_config.get('weight_decay')),
            adam_beta2=0.999,
            logging_steps=100,
            optim=model_config.get('optimizer'), # you can use paged optimizers like paged_adamw_8bit for QLoRA
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=1,
            output_dir=output_dir,
            bf16=True,
            report_to="wandb",
            dataloader_pin_memory=False,
            label_names=["labels"]
        )

trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=clevrer_collate_fn,
        args=args
        )

trainer.train()