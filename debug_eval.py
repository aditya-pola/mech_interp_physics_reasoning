import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import sys
import torch
import yaml
import json
from transformers import TrainingArguments
from peft import PeftModel

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    home_dir = os.environ.get("HOME_DIR", os.path.abspath(os.path.join(script_dir, ".")))
    HOME_DIR = home_dir
    sys.path.insert(0, HOME_DIR)

    # Load config
    config_path = os.path.join(home_dir, "base_eval_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    from src.processing_paligemma import PaliGemmaProcessor
    from src.modeling_paligemma import PaliGemmaForConditionalGeneration
    from src.utils import make_clevrer_collate_fn
    from src.data import ClevrerDataset

    model_config = config["model_train"]
    data_config = config["data_config"]
    model_id = model_config["model"]

    # Create dataset
    test_frames_dir = os.path.join(HOME_DIR, data_config.get("data_path", "test_frames"))
    annotations_path = os.path.join(HOME_DIR, data_config.get("json_path", "miscellaneous/validation.json"))

    dataset = ClevrerDataset(
        frames_root=test_frames_dir,
        json_path=annotations_path,
        question_type=data_config.get("question_type", "all"),
        transform=None,
        shuffle=False
    )

    print(f"Total samples in dataset: {len(dataset)}")
    
    # Check test size
    test_size = data_config['test_size']
    print(f"Test size from config: {test_size}")
    
    # Split dataset
    _, test_ds = dataset.train_test_split(
        test_size=test_size,
        cache_path="debug_split.json"
    )
    
    print(f"Test dataset size: {len(test_ds)}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="eager",
        token_compression=model_config.get('token_compression')
    )
    
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    # Test with a single sample
    if len(test_ds) > 0:
        sample = test_ds[0]
        print(f"\nTesting with first sample:")
        print(f"Question: {sample['question']}")
        print(f"Expected answer: {sample['answer']}")
        print(f"Question type: {sample['question_type']}")
        
        # Create collate function
        collate_fn = make_clevrer_collate_fn(
            model=model,
            processor=processor,
            model_config=model_config,
            data_config=data_config,
            dtype=model.dtype
        )
        
        # Process single sample
        batch = collate_fn([sample])
        
        # Generate prediction
        with torch.no_grad():
            generated_tokens = model.generate(
                **batch,
                max_new_tokens=100,
                do_sample=False,
            )
        
        input_len = batch["input_ids"].shape[-1]
        pred_tokens = generated_tokens[:, input_len:]
        
        print(f"\nGenerated token IDs: {pred_tokens}")
        
        # Decode tokens
        decoded = processor.batch_decode(pred_tokens, skip_special_tokens=True)
        print(f"Decoded prediction: {decoded}")
        
        # Also decode without skipping special tokens to see what's being generated
        decoded_with_special = processor.batch_decode(pred_tokens, skip_special_tokens=False)
        print(f"Decoded with special tokens: {decoded_with_special}")
        
        # Check if prediction is empty after filtering special tokens
        special_tokens = {0, 1, 2, 3, -100, 257152}
        filtered_pred = [x.item() for x in pred_tokens[0] if x.item() not in special_tokens]
        print(f"Filtered prediction tokens: {filtered_pred}")
        
        if len(filtered_pred) == 0:
            print("\nWARNING: Model generated only special tokens!")

if __name__ == "__main__":
    main()
