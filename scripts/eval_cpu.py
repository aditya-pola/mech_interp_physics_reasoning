import os
# Remove CUDA device setting for CPU-only mode
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import sys
import torch
from datetime import datetime
import argparse
import yaml
import numpy as np
import json
from transformers import TrainingArguments
from peft import PeftModel
from collections import defaultdict
import glob


def get_device_map():
    # Force CPU mode
    return "cpu"

def main():
    parser = argparse.ArgumentParser(description="Evaluate a PaliGemma model (base or with LoRA) on CLEVRER test set.")
    parser.add_argument("checkpoint_dir", type=str, nargs="?", help="Relative path to checkpoint folder from HOME_DIR")
    parser.add_argument("--base", action="store_true", help="Evaluate base model without LoRA adapter")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    home_dir = os.environ.get("HOME_DIR", os.path.abspath(os.path.join(script_dir, "..")))
    HOME_DIR = home_dir
    sys.path.insert(0, HOME_DIR)

    if args.base:
        config_path = os.path.join(home_dir, "base_eval_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Base config file not found: {config_path}")
    else:
        if not args.checkpoint_dir:
            raise ValueError("checkpoint_dir must be provided unless --base is used")
        checkpoint_dir = os.path.join(home_dir, args.checkpoint_dir)
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.yaml not found in {checkpoint_dir}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    from src.processing_paligemma import PaliGemmaProcessor
    from src.modeling_paligemma import PaliGemmaForConditionalGeneration
    from src.utils import make_clevrer_collate_fn, compute_accuracy, CLEVRERTrainer
    from src.data import ClevrerDataset

    model_config = config["model_train"]
    data_config = config["data_config"]
    model_id = model_config["model"]
    question_type = data_config.get("question_type", "all")

    test_frames_dir = os.path.join(HOME_DIR, data_config.get("data_path", "test_frames"))
    annotations_path = os.path.join(HOME_DIR, data_config.get("json_path", "miscellaneous/validation.json"))

    if not os.path.exists(test_frames_dir):
        raise FileNotFoundError(f"Test frames directory not found: {test_frames_dir}")
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Validation annotations not found: {annotations_path}")

    dataset = ClevrerDataset(
        frames_root=test_frames_dir,
        json_path=annotations_path,
        question_type=question_type,
        transform=None,
        shuffle=False
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.base:
        results_root = os.path.join(HOME_DIR, "artifacts", "BASE", f"eval_{question_type}_{timestamp}")
    else:
        results_root = os.path.join(checkpoint_dir, f"eval_{question_type}_{timestamp}")

    os.makedirs(results_root, exist_ok=True)
    split_cache_path = os.path.join(results_root, "split_indices.json")

    train_ds, test_ds = dataset.train_test_split(
        test_size=data_config['test_size'],
        cache_path=split_cache_path
    )

    if not args.base:
        last_checkpoint = None
        if os.path.isdir(checkpoint_dir):
            checkpoints = list(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")))
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
                last_checkpoint = checkpoints[-1]

    device = get_device_map()
    print(f"Using device: {device}")
    print("WARNING: Running on CPU will be VERY slow. Consider using cloud GPU services.")
    
    if args.base:
        print("Loading base model only (no LoRA).")
        # Load with CPU and potentially quantization
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,  # Use float32 for CPU
            attn_implementation="eager",
            token_compression=model_config.get('token_compression')
        )
    else:
        print(f"Loading model with LoRA adapter from: {checkpoint_dir}")
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            attn_implementation="eager",
            token_compression=model_config.get("token_compression"),
            target_length=model_config.get("target_length")
        )
        model = PeftModel.from_pretrained(base_model, last_checkpoint)

    results_file = os.path.join(results_root, "eval_results.txt")
    details_path = os.path.join(results_root, "eval_details.json")

    processor = PaliGemmaProcessor.from_pretrained(model_id)
    collate_fn = make_clevrer_collate_fn(
        model=model,
        processor=processor,
        model_config=model_config,
        data_config=data_config,
        dtype=model.dtype
    )

    eval_args = TrainingArguments(
        output_dir="/tmp/eval",
        per_device_eval_batch_size=1,  # Reduced for CPU
        eval_accumulation_steps=1,
        dataloader_pin_memory=False,
        fp16=False,  # Disable for CPU
        bf16=False,  # Disable for CPU
        remove_unused_columns=False,
        report_to=[],
        save_strategy="no",
        logging_strategy="no"
    )

    trainer = CLEVRERTrainer(
        model=model,
        args=eval_args,
        eval_dataset=test_ds,
        data_collator=collate_fn,
        compute_metrics=compute_accuracy,
        processing_class=processor
    )

    # For CPU, maybe evaluate on fewer samples
    if len(test_ds) > 10:
        print(f"NOTE: Evaluating on first 10 samples only due to CPU constraints")
        test_ds_subset = torch.utils.data.Subset(test_ds, range(10))
        pred_output = trainer.predict(test_ds_subset)
    else:
        pred_output = trainer.predict(test_ds)
        
    preds = pred_output.predictions
    labels = pred_output.label_ids

    preds = preds.tolist()
    labels = labels.tolist()

    special_tokens = {0, 1, 2, 3, -100, 257152}
    correct_flags = []
    per_sample_results = []
    type_correct = defaultdict(int)
    type_total = defaultdict(int)

    for i, (pred_row, label_row) in enumerate(zip(preds, labels)):
        item = test_ds[i] if i < len(test_ds) else test_ds_subset[i]
        qtype = item["question_type"]

        filtered_pred = [x for x in pred_row if x not in special_tokens]
        filtered_label = [x for x in label_row if x not in special_tokens]

        correct = sorted(filtered_pred) == sorted(filtered_label)
        correct_flags.append(correct)

        type_total[qtype] += 1
        type_correct[qtype] += int(correct)

        per_sample_results.append({
            "question_type": qtype,
            "question_id": item["question_id"],
            "video_filename": item["video_filename"],
            "predicted_token_ids": filtered_pred,
            "label_token_ids": filtered_label,
            "correct": correct
        })

    with open(details_path, "w") as f:
        json.dump(per_sample_results, f, indent=2)

    accuracy = np.mean(correct_flags) if correct_flags else 0.0
    with open(results_file, "w") as f:
        f.write(f"accuracy: {accuracy:.4f}\n\n")
        f.write("Question Type Accuracies:\n")
        for qtype, total in type_total.items():
            acc = type_correct[qtype] / total if total > 0 else 0.0
            f.write(f"{qtype}: {acc:.4f} ({type_correct[qtype]}/{total})\n")

    print(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    print(f"Results directory: {results_root}")


if __name__ == "__main__":
    main()
