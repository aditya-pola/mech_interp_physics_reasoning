# ------------------ Show available CUDA devices ------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import sys
import torch
from datetime import datetime
import argparse
import yaml
import numpy as np
import json
from transformers import TrainingArguments
from peft import PeftModel


def get_device_map():
    return "cuda" if torch.cuda.is_available() else "cpu"


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

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    from src.processing_paligemma import PaliGemmaProcessor
    from src.modeling_paligemma import PaliGemmaForConditionalGeneration
    from src.utils import make_clevrer_collate_fn, compute_accuracy, CLEVRERTrainer
    from src.data import ClevrerDataset

    model_config = config["model_train"]
    data_config = config["data_config"]
    model_id = model_config["model"]

    # Dataset setup
    test_frames_dir = os.path.join(HOME_DIR, data_config.get("data_path", "test_frames"))
    annotations_path = os.path.join(HOME_DIR, data_config.get("json_path", "miscellaneous/validation.json"))

    if not os.path.exists(test_frames_dir):
        raise FileNotFoundError(f"Test frames directory not found: {test_frames_dir}")
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Validation annotations not found: {annotations_path}")

    dataset = ClevrerDataset(
        frames_root=test_frames_dir,
        json_path=annotations_path,
        transform=None
    )

    train_ds, _ = dataset.train_test_split(test_size=0.0)
    test_ds = train_ds

    # Load model
    device = get_device_map()
    if args.base:
        print("Loading base model only (no LoRA).")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            attn_implementation="eager",
            token_compression=model_config.get('token_compression')
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(HOME_DIR, "miscellaneous", f"eval_base_{timestamp}.txt")
    else:
        print(f"Loading model with LoRA adapter from: {checkpoint_dir}")
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            attn_implementation="eager",
            token_compression=model_config.get("token_compression"),
            target_length=model_config.get("target_length")
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model.to(device)
        results_file = os.path.join(checkpoint_dir, "eval_results.txt")

    # Load processor and collate_fn
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    collate_fn = make_clevrer_collate_fn(
        processor=processor,
        model_config=model_config,
        data_config=data_config,
        dtype=model.dtype
    )

    # Evaluation args (no output or logging)
    eval_args = TrainingArguments(
        output_dir="/tmp/eval",
        per_device_eval_batch_size=model_config.get("eval_batch_size", 8),
        dataloader_pin_memory=False,
        bf16=True,
        remove_unused_columns=False,
        report_to=[],
        save_strategy="no",
        logging_strategy="no"
    )

    # Run prediction
    trainer = CLEVRERTrainer(
        model=model,
        args=eval_args,
        eval_dataset=test_ds,
        data_collator=collate_fn,
        compute_metrics=compute_accuracy,
        processing_class=processor
    )

    pred_output = trainer.predict(test_ds)
    preds = pred_output.predictions
    labels = pred_output.label_ids

    # Decode results
    preds = preds.tolist()
    labels = labels.tolist()

    per_sample_results = []
    correct_flags = []

    for i, (pred_id, (label_id, qtype_id)) in enumerate(zip(preds, labels)):
        item = test_ds[i]
        correct = pred_id == label_id
        correct_flags.append(correct)

        per_sample_results.append({
            "question_id": item["question_id"],
            "video_filename": item["video_filename"],
            "question_type_id": int(qtype_id),
            "predicted_token_id": int(pred_id),
            "label_token_id": int(label_id),
            "correct": correct
        })

    # Save detailed results
    results_root = HOME_DIR if args.base else checkpoint_dir
    details_path = os.path.join(results_root, "eval_details.json")
    with open(details_path, "w") as f:
        json.dump(per_sample_results, f, indent=2)

    # Save accuracy
    accuracy = np.mean(correct_flags)
    with open(results_file, "w") as f:
        f.write(f"accuracy: {accuracy:.4f}\n")

    print(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    print(f"Per-sample results saved to: {details_path}")


if __name__ == "__main__":
    main()
