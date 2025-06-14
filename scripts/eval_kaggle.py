import os
# Use GPU 0 in Kaggle (typically)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
    return "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="Evaluate a PaliGemma model on CLEVRER test set on Kaggle.")
    parser.add_argument("checkpoint_dir", type=str, nargs="?", help="Relative path to checkpoint folder (within Kaggle input dataset if used)")
    parser.add_argument("--base", action="store_true", help="Evaluate base model without LoRA")
    args = parser.parse_args()

    # In Kaggle, the script itself will be in /kaggle/input/your-code-dataset/scripts/
    # HOME_DIR should point to /kaggle/input/your-code-dataset/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    home_dir = os.environ.get("HOME_DIR", os.path.abspath(os.path.join(script_dir, "..")))
    HOME_DIR = home_dir # This is the project root, e.g., /kaggle/input/mech-interp-project-code/
    sys.path.insert(0, HOME_DIR)

    if args.base:
        # Assumes base_eval_config.yaml is at the root of the code dataset
        config_path = os.path.join(HOME_DIR, "base_eval_config.yaml")
    else:
        if not args.checkpoint_dir:
            raise ValueError("checkpoint_dir must be provided unless --base is used")
        # Checkpoint_dir is relative to HOME_DIR (e.g. "artifacts/my_checkpoint")
        # or an absolute path if it's in a different Kaggle dataset.
        # For simplicity, let's assume it's relative to HOME_DIR for now.
        checkpoint_dir_abs = os.path.join(HOME_DIR, args.checkpoint_dir)
        config_path = os.path.join(checkpoint_dir_abs, "config.yaml")

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

    # Use absolute paths from the modified config (paths will point to /kaggle/input/your-data-dataset/)
    test_frames_dir = data_config["data_path"]
    annotations_path = data_config["json_path"]

    print(f"Loading dataset from: {test_frames_dir}")
    print(f"Annotations from: {annotations_path}")

    dataset = ClevrerDataset(
        frames_root=test_frames_dir,
        json_path=annotations_path,
        question_type=question_type,
        transform=None,
        shuffle=False
    )

    print(f"Total samples in dataset: {len(dataset)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    KAGGLE_WORKING_DIR = "/kaggle/working/"

    if args.base:
        results_dir_name = os.path.join("artifacts", "BASE", f"eval_{question_type}_{timestamp}")
    else:
        # Assuming checkpoint_dir was relative to HOME_DIR
        checkpoint_name = os.path.basename(os.path.normpath(args.checkpoint_dir))
        results_dir_name = os.path.join("artifacts", checkpoint_name, f"eval_{question_type}_{timestamp}")
    
    results_root = os.path.join(KAGGLE_WORKING_DIR, results_dir_name)
    os.makedirs(results_root, exist_ok=True)
    
    # split_cache_path will also be in /kaggle/working/
    split_cache_path = os.path.join(results_root, "split_indices.json")


    train_ds, test_ds = dataset.train_test_split(
        test_size=data_config['test_size'], # This could be a number or a float
        cache_path=split_cache_path
    )

    print(f"Test dataset size: {len(test_ds)}")

    device = get_device_map()
    print(f"Using device: {device}")
    
    print("Loading model...")
    if args.base:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto", # Changed from device for multi-GPU, auto is safer
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="eager", # or "flash_attention_2" if available and preferred
            token_compression=model_config.get('token_compression')
        )
    else:
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="eager",
            token_compression=model_config.get("token_compression"),
            target_length=model_config.get("target_length")
        )
        # last_checkpoint path needs to be relative to HOME_DIR or absolute
        last_checkpoint_path = os.path.join(checkpoint_dir_abs, f"checkpoint-{config['training_args']['num_train_epochs']}") # Example, adjust as needed
        # Or, if you save multiple checkpoints:
        # all_checkpoints = glob.glob(os.path.join(checkpoint_dir_abs, "checkpoint-*"))
        # if not all_checkpoints:
        #     raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir_abs}")
        # last_checkpoint_path = sorted(all_checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        
        # For simplicity, assuming a specific checkpoint or the last one based on config.
        # If using glob, ensure checkpoint_dir_abs is correct.
        # The original glob was: glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))[-1]
        # This needs to be adapted based on where checkpoint_dir (relative) points.
        # If args.checkpoint_dir is "artifacts/my_checkpoint", then checkpoint_dir_abs is /kaggle/input/code/artifacts/my_checkpoint
        
        # Let's use a robust way to find the last checkpoint if multiple exist
        all_ckpts = glob.glob(os.path.join(checkpoint_dir_abs, "checkpoint-*"))
        if not all_ckpts:
             raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir_abs}. Ensure checkpoint_dir argument is correct.")
        last_checkpoint = max(all_ckpts, key=lambda p: int(p.split("-")[-1]))
        print(f"Loading PEFT model from checkpoint: {last_checkpoint}")
        model = PeftModel.from_pretrained(base_model, last_checkpoint)


    print("Model loaded successfully!")

    processor = PaliGemmaProcessor.from_pretrained(model_id)
    collate_fn = make_clevrer_collate_fn(
        model=model,
        processor=processor,
        model_config=model_config,
        data_config=data_config,
        dtype=model.dtype
    )

    eval_args = TrainingArguments(
        output_dir=os.path.join(KAGGLE_WORKING_DIR, "tmp_eval_output"), # Temporary, ensure it's writable
        per_device_eval_batch_size=model_config.get("eval_batch_size", 4),
        eval_accumulation_steps=1,
        dataloader_pin_memory=False, # True might be better if GPU memory allows
        bf16=True if device == "cuda" else False,
        remove_unused_columns=False,
        report_to=[], # No reporting needed for simple eval
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

    print("Starting evaluation...")
    pred_output = trainer.predict(test_ds)
    
    preds = pred_output.predictions.tolist()
    labels = pred_output.label_ids.tolist()

    special_tokens = {0, 1, 2, 3, -100, 257152} # Common special tokens, adjust if needed

    progressive_details_path = os.path.join(results_root, "eval_details_progressive.json")
    progressive_summary_path = os.path.join(results_root, "eval_summary_progressive.txt")

    all_per_sample_results = []
    running_correct_flags = []
    running_type_correct = defaultdict(int)
    running_type_total = defaultdict(int)

    print(f"Starting processing of {len(preds)} predictions. Results will be saved to {results_root}")

    for i, (pred_row, label_row) in enumerate(zip(preds, labels)):
        # Ensure test_ds[i] is valid; if test_ds was shuffled, direct indexing might be tricky
        # Assuming test_ds maintains order or provides necessary metadata directly
        item = test_ds[i] # Simpler and standard way to get item from PyTorch Dataset/Subset
        qtype = item["question_type"]

        filtered_pred = [x for x in pred_row if x not in special_tokens]
        filtered_label = [x for x in label_row if x not in special_tokens]
        correct = sorted(filtered_pred) == sorted(filtered_label)
        
        current_sample_result_data = {
            "question_type": qtype,
            "question_id": item["question_id"],
            "video_filename": item["video_filename"],
            "predicted_token_ids": filtered_pred,
            "label_token_ids": filtered_label,
            "correct": correct
        }
        all_per_sample_results.append(current_sample_result_data)

        running_correct_flags.append(correct)
        running_type_total[qtype] += 1
        running_type_correct[qtype] += int(correct)

        if (i + 1) % 5 == 0 or (i + 1) == len(preds):
            with open(progressive_details_path, "w") as f_details:
                json.dump(all_per_sample_results, f_details, indent=2)

            current_overall_accuracy = np.mean(running_correct_flags) if running_correct_flags else 0.0
            
            with open(progressive_summary_path, "w") as f_summary:
                f_summary.write(f"Processed samples: {len(all_per_sample_results)} / {len(preds)}\n")
                f_summary.write(f"Overall Accuracy (so far): {current_overall_accuracy:.4f}\n\n")
                f_summary.write("Question Type Accuracies (so far):\n")
                for qt_summary, total_summary in running_type_total.items():
                    acc_summary = running_type_correct[qt_summary] / total_summary if total_summary > 0 else 0.0
                    f_summary.write(f"{qt_summary}: {acc_summary:.4f} ({running_type_correct[qt_summary]}/{total_summary})\n")
            
            print(f"Progress: Saved results for {len(all_per_sample_results)} samples. Accuracy so far: {current_overall_accuracy:.4f}")

    final_accuracy = np.mean(running_correct_flags) if running_correct_flags else 0.0
    print(f"\n{'='*50}")
    print(f"Evaluation Complete!")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {final_accuracy:.4f}")
    print(f"\nQuestion Type Accuracies:")
    for qtype_final, total_final in running_type_total.items():
        acc_final = running_type_correct[qtype_final] / total_final if total_final > 0 else 0.0
        print(f"  {qtype_final}: {acc_final:.4f} ({running_type_correct[qtype_final]}/{total_final})")
    print(f"\nFinal results saved to directory: {results_root}")
    print(f"  Detailed progressive results: {progressive_details_path}")
    print(f"  Summary progressive results: {progressive_summary_path}")

if __name__ == "__main__":
    main()
