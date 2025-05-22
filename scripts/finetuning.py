import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

import torch
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from transformers import TrainingArguments, Trainer
from huggingface_hub import login as hf_login
import yaml
import sys
from datetime import datetime
import pytz
import glob
import shutil

# ----------------------------------
IS_TEST_RUN = True # Set to False for actual training runs
# ----------------------------------

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

HOME_DIR = config.get('HOME')
sys.path.insert(0, HOME_DIR)

from src.processing_paligemma import PaliGemmaProcessor
from src.modeling_paligemma import PaliGemmaForConditionalGeneration
from src.utils import lora_filter, freeze_vision_tower, load_dataset, \
                      generate_run_details, create_valid_dirname, \
                      CLEVRERTrainer, compute_accuracy, setup_wandb

ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

model_config = config.get('model_train', {})
lora_config_params = config.get('lora', {})
data_config = config.get('data_config', {})


if IS_TEST_RUN:
    run_name = f"TEST_RUN_{now_ist.strftime('%Y%m%d_%H%M%S')}"
    print(f"IS_TEST_RUN is True. Test run name: {run_name}. WandB logging skipped.")
else:
    run_name, _, wandb_config_for_run = generate_run_details(config)
    print(f"WandB Run Name and Directory Name ---> {run_name}")

    setup_wandb(project_name="Physical_Reasoning",
                run_name=run_name,
                config_dict=wandb_config_for_run,
                key="c05e9a6ff01ac9550c6c83b7c666c67f0d68872")

hf_login(token="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")

data_config['HOME'] = HOME_DIR
train_ds, val_ds = load_dataset(data_config)

model_id = model_config.get('model')
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id,
                                                        #   device_map="auto",
                                                          device_map=device,
                                                          attn_implementation="eager",
                                                          token_compression=model_config.get('token_compression'),
                                                          target_length=model_config.get('target_length'))

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
    prompts, labels_text, images, question_types_list = [], [], [], []

    for example in examples:
        if example['question_type'] != 'descriptive':
            text = "Select all that apply. " + example["question"]
        else:
            text = "Answer with one word or number only. " + example["question"]

        num_image_tokens_calc = (model_config.get('target_length', 128) * data_config.get('num_frames', 8)) / 1024
        image_tokens_count = int(num_image_tokens_calc)

        image_tokens = "<image> " * image_tokens_count
        prompt = image_tokens + text + " en"

        prompts.append(prompt)
        labels_text.append(example['answer'])
        images.append(example['frames'])
        question_types_list.append(example['question_type'])

    tokens = processor(
        text=prompts,
        images=images,
        return_tensors='pt',
        do_rescale=False,
        padding="longest",
        suffix=labels_text,
    ).to(DTYPE)

    tokens["question_types"] = question_types_list

    return tokens

if not IS_TEST_RUN:
    valid_dirname = create_valid_dirname(run_name)
    output_dir = os.path.join(HOME_DIR, model_config.get('save_dir'), valid_dirname)
    os.makedirs(output_dir, exist_ok=True)

    split_indices_src = os.path.join("miscellaneous", "split_indices.json")
    split_indices_dst = os.path.join(output_dir, "split_indices.json")
    if os.path.exists(split_indices_src):
        shutil.move(split_indices_src, split_indices_dst)
        print(f"Moved {split_indices_src} to {split_indices_dst}")
    else:
        print(f"Warning: {split_indices_src} not found.")

    # Copy config.yaml to output_dir
    config_src = os.path.join(HOME_DIR, "config.yaml")
    config_dst = os.path.join(output_dir, "config.yaml")
    if os.path.exists(config_src):
        shutil.copy2(config_src, config_dst)
        print(f"Copied {config_src} to {config_dst}")
    else:
        print(f"Warning: {config_src} not found.")

else:
    output_dir = None
    split_indices_path_to_delete = os.path.join(HOME_DIR, "miscellaneous", "split_indices.json")
    if os.path.exists(split_indices_path_to_delete):
        os.remove(split_indices_path_to_delete)

report_to_value = "none" if IS_TEST_RUN else "wandb"

args=TrainingArguments(
            num_train_epochs=model_config.get('num_epochs', 3),
            remove_unused_columns=False,
            per_device_train_batch_size=model_config.get('batch_size'),
            per_device_eval_batch_size=model_config.get('eval_batch_size', 8),
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=float(model_config.get('learning_rate')),
            weight_decay=float(model_config.get('weight_decay')),
            adam_beta2=0.999,
            logging_steps=100,
            optim=model_config.get('optimizer'),
            bf16=True,
            report_to=report_to_value,
            dataloader_pin_memory=False,
            label_names=["labels"],
            eval_strategy="steps",
            eval_steps=model_config.get('eval_steps', 500),
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_strategy="steps",
            save_steps=model_config.get('save_steps', 500),
            save_total_limit=model_config.get('save_total_limit', 2),
            output_dir=output_dir,
        )

trainer = CLEVRERTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=clevrer_collate_fn,
        args=args,
        processing_class=processor,
        compute_metrics=compute_accuracy,
        )

last_checkpoint = None
if os.path.isdir(args.output_dir):
    checkpoints = list(glob.glob(os.path.join(args.output_dir, "checkpoint-*")))
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        last_checkpoint = checkpoints[-1]
        print(f"Resuming from checkpoint: {last_checkpoint}")
    else:
        print(f"No checkpoint found in {args.output_dir}. Starting fresh.")
else:
    print(f"Output directory {args.output_dir} does not exist. Starting fresh.")

trainer.train(resume_from_checkpoint=last_checkpoint)