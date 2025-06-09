import wandb
import os
from torchvision import transforms
from src.data import ClevrerDataset
from datetime import datetime
import pytz
import re
from collections import defaultdict
import numpy as np

from transformers import Trainer
import torch

def load_dataset(data_config):
    """
    Loads and splits the ClevrerDataset based on the provided data configuration.

    Args:
        data_config (dict): A dictionary containing dataset configuration parameters
                            like 'image_size', 'data_path', 'json_path',
                            'question_type', 'num_frames', and 'test_size'.

    Returns:
        tuple: A tuple containing (train_ds, val_ds) after splitting the dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((data_config.get('image_size'), data_config.get('image_size'))),
        transforms.ToTensor(),
    ])

    HOME_DIR = data_config['HOME']

    data_path = os.path.join(HOME_DIR, data_config['data_path'])
    json_path = os.path.join(HOME_DIR, data_config['json_path'])

    dataset = ClevrerDataset(
        frames_root=data_path,
        json_path=json_path,
        transform=transform,
        question_type=data_config.get('question_type'),
        NUM_FRAMES=data_config.get('num_frames')
    )

    train_ds, val_ds = dataset.train_test_split(test_size=data_config.get('test_size'), random_seed=data_config.get('seed', 42))
    return train_ds, val_ds

def generate_run_details(base_config, artists_file="miscellaneous/artists.txt", used_artists_file="miscellaneous/used_artists.txt"):
    """
    Generates a dynamic run name string and selects an artist name for WandB.
    It also handles moving the artist name from artists.txt to used_artists.txt.

    Args:
        base_config (dict): The main configuration dictionary for your training run.
        artists_file (str): Path to the file containing available artist names.
        used_artists_file (str): Path to the file to log used artist names.

    Returns:
        tuple: (artist_name (str), dynamic_run_name (str), wandb_config_dict (dict))
               - artist_name: The chosen artist name for the WandB run.
               - dynamic_run_name: The detailed dynamic run name string.
               - wandb_config_dict: A dictionary of key training parameters to log to WandB.
    """
    now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))

    # --- Generate Dynamic Run Name (Detailed) ---
    full_model_name = base_config['model_train']['model']
    model_name_short = "PaliG" if full_model_name == "google/paligemma2-3b-mix-448" else full_model_name.split('/')[-1]
    
    num_epochs = base_config['model_train']['num_epochs']
    batch_size = base_config['model_train']['batch_size']
    learning_rate = base_config['model_train']['learning_rate']
    image_size = base_config['data_config']['image_size']
    num_frames = base_config['data_config']['num_frames']

    token_compression = base_config['model_train']['token_compression']
    target_length = base_config['model_train']['target_length']

    lora_config = base_config['lora']
    lora_rank = lora_config['rank']
    lora_layers = lora_config['layers']
    lora_layer_types = lora_config.get('layer_types', [])
    lora_include_vision = lora_config['include_vision']
    lora_include_language = lora_config['include_language']
    lora_vision_layers = lora_config.get('vision_layers')
    lora_language_layers = lora_config.get('language_layers')

    layer_type_map = {
        "self_attn": "attn", "mlp": "mlp", "embeddings": "emb", "projector": "proj",
        "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"
    }
    succinct_layer_types = [layer_type_map.get(lt, lt) for lt in lora_layer_types]
    layer_types_str = "_".join(succinct_layer_types) if succinct_layer_types else "all_types"

    layers_str = ""
    if lora_vision_layers is not None and lora_language_layers is not None:
        layers_str = "selected_layers"
    elif lora_vision_layers is not None:
        layers_str = "vision_selected"
    elif lora_language_layers is not None:
        layers_str = "lang_selected"
    elif lora_layers == "all":
        layers_str = "all_layers"
    else:
        layers_str = "selected_layers"

    inclusion_mode = ""
    if lora_include_vision and lora_include_language:
        inclusion_mode = "vision_and_language"
    elif lora_include_vision and not lora_include_language:
        inclusion_mode = "only_vision"
    elif not lora_include_vision and lora_include_language:
        inclusion_mode = "only_language"
    else:
        inclusion_mode = "no_vision_no_language"

    dynamic_run_name_parts = [
        now_ist.strftime("%m-%d_%H:%M"),
        model_name_short,
        f"frames={num_frames}",
        f"lora: r={lora_rank}",
        inclusion_mode,
        f"{layers_str}->{layer_types_str}",
        f"{token_compression}-compr_{target_length}toks", # Simplified from 1024->{target_length}toks
        f"e={num_epochs}",
        f"b={batch_size}",
        f"lr={learning_rate}",
        f"im_size={image_size}",
    ]
    dynamic_run_name = " ".join(dynamic_run_name_parts)

    # --- Artist Name Logic ---
    artist_name = "default_artist" # Fallback
    try:
        with open(artists_file, 'r') as f:
            artists = [line.strip() for line in f if line.strip()]

        if not artists:
            print(f"Warning: {artists_file} is empty. Using a default artist name.")
            artist_name = f"default_artist_{now_ist.strftime('%H%M%S')}" # Add timestamp for uniqueness
        else:
            # Pick a random artist, or simply the first one
            artist_name = artists.pop(0) # Take the first artist and remove it
            print(f"Selected artist for run: {artist_name}")

            # Write remaining artists back to file
            with open(artists_file, 'w') as f:
                for artist in artists:
                    f.write(artist + '\n')

            # Append used artist to used_artists.txt
            with open(used_artists_file, 'a') as f:
                f.write(f"{artist_name} --- {dynamic_run_name}\n")

    except FileNotFoundError:
        print(f"Warning: {artists_file} not found. Using a default artist name.")
        artist_name = f"default_artist_{now_ist.strftime('%H%M%S')}"
    except Exception as e:
        print(f"Error handling artist files: {e}. Using a default artist name.")
        artist_name = f"default_artist_{now_ist.strftime('%H%M%S')}"

    # --- Construct WandB Config Dictionary ---
    wandb_config_dict = {
        "model_name": full_model_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "num_frames": num_frames,
        "token_compression": token_compression,
        "target_length": target_length,
        "lora_rank": lora_rank,
        "lora_layers_scope": lora_layers,
        "lora_layer_types": lora_layer_types,
        "lora_include_vision": lora_include_vision,
        "lora_include_language": lora_include_language,
        "lora_vision_layers": lora_vision_layers,
        "lora_language_layers": lora_language_layers,
        "start_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
    }

    return artist_name, dynamic_run_name, wandb_config_dict

def create_valid_dirname(run_name):
    """
    Converts a given string into a valid Linux directory name.

    Replaces problematic characters (spaces, colons, slashes, etc.) with underscores
    and removes any leading/trailing unsafe characters.

    Args:
        run_name (str): The original run name string.

    Returns:
        str: A valid Linux directory name.
    """
    # Replace spaces with underscores
    dirname = run_name.replace(" ", "_")
    # Replace colons (:) with hyphens or remove, as they can be problematic in some contexts
    dirname = dirname.replace(":", "-")
    # Replace slashes (/) with underscores
    dirname = dirname.replace("/", "_")
    # Remove any characters that are not alphanumeric, underscore, or hyphen
    dirname = re.sub(r'[^\w.-]', '', dirname) # Keeps letters, numbers, underscore, dot, hyphen
    # Remove multiple consecutive underscores
    dirname = re.sub(r'_{2,}', '_', dirname)
    # Remove leading/trailing underscores or hyphens
    dirname = dirname.strip('_-')
    # Ensure it's not empty
    if not dirname:
        dirname = "untitled_run"
    return dirname

def setup_wandb(project_name, run_name, config_dict, key):
    """
    Sets up WandB logging for the training run.

    Args:
        project_name (str): The name of the WandB project.
        run_name (str): The name for this specific WandB run (e.g., artist name).
        config_dict (dict): A dictionary of parameters to log as the run's config.
        key (str): Your WandB API key.
    """
    try:
        wandb.login(key=key)
        print("Successfully logged into WandB.")
    except Exception as e:
        print(f"Error logging into WandB: {e}")

    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "true"

    try:
        # Pass the config_dict directly to the config argument
        wandb.init(project=project_name, name=run_name, config=config_dict, resume="allow")
        print(f"WandB run initialized: Project - {project_name}, Run - {run_name}")
    except Exception as e:
        print(f"Error initializing WandB run: {e}")

def freeze_vision_tower(model):
    """
    Freezes the parameters of the vision tower in a multi-modal model.
    Args:
        model (nn.Module): The multi-modal model.
    """
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False
            # print(f"Froze parameter: {name}")
        else:
            pass
            # print(f"Parameter {name} remains trainable.")

def lora_filter(model, layers="all", layer_types=None, include_vision=True, include_language=True, vision_layers=None, language_layers=None):
    """
    Filters and formats parameter names for LoRA application from a PyTorch model.

    Args:
        model: The PyTorch model.
        layers (str or list, optional): Specifies which layers to consider.
            If "all", considers all layers in included towers and projector.
            If a list, it acts as a global layer index list for both towers
            if vision_layers and language_layers are None. Defaults to "all".
        layer_types (list, optional): A list of layer types to include.
            Supported types: "self_attn", "mlp", "embeddings", "projector",
            "q_proj", "k_proj", "v_proj", "o_proj". If None or empty, no
            specific layer type filtering is applied. Defaults to None.
        include_vision (bool, optional): Whether to include parameters from the
            vision tower. Defaults to True.
        include_language (bool, optional): Whether to include parameters from the
            language model. Defaults to True.
        vision_layers (list, optional): A list of specific layer indices to
            consider within the vision tower. If provided, overrides the
            integer elements in the 'layers' parameter for the vision tower.
            Defaults to None.
        language_layers (list, optional): A list of specific layer indices to
            consider within the language model. If provided, overrides the
            integer elements in the 'layers' parameter for the language model.
            Defaults to None.

    Returns:
        list: A list of filtered and formatted parameter names, in the
              order they appear in the model's named_parameters.
    """
    filtered_names = []
    num_vision_layers = 26
    num_language_layers = 25

    considered_vision_layers = []
    considered_language_layers = []

    if include_vision:
        if vision_layers is not None:
            considered_vision_layers = [l for l in vision_layers if isinstance(l, int) and 0 <= l < num_vision_layers]
        elif layers == "all":
            considered_vision_layers = list(range(num_vision_layers))
        elif isinstance(layers, list):
            considered_vision_layers = [l for l in layers if isinstance(l, int) and 0 <= l < num_vision_layers]

    if include_language:
        if language_layers is not None:
            considered_language_layers = [l for l in language_layers if isinstance(l, int) and 0 <= l < num_language_layers]
        elif layers == "all":
            considered_language_layers = list(range(num_language_layers))
        elif isinstance(layers, list):
            considered_language_layers = [l for l in layers if isinstance(l, int) and 0 <= l < num_language_layers]

    include_self_attn = layer_types and "self_attn" in layer_types
    include_mlp = layer_types and "mlp" in layer_types
    include_embeddings = layer_types and "embeddings" in layer_types
    include_projector = layer_types and "projector" in layer_types
    include_q_proj = layer_types and "q_proj" in layer_types and not include_self_attn
    include_k_proj = layer_types and "k_proj" in layer_types and not include_self_attn
    include_v_proj = layer_types and "v_proj" in layer_types and not include_self_attn
    include_o_proj = layer_types and "o_proj" in layer_types and not include_self_attn

    for name, _ in model.named_parameters():
        if ".bias" in name or "layer_norm" in name:
            continue

        if include_vision and "vision_tower" in name:
            layer_match = False
            if layers == "all" or (isinstance(vision_layers, list) and any(f".layers.{layer_index}." in name for layer_index in considered_vision_layers)) or (vision_layers is None and isinstance(layers, list) and any(f".layers.{layer_index}." in name for layer_index in considered_vision_layers)):
                layer_match = True

            if layer_match:
                include = False
                if include_self_attn and "self_attn" in name and any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                    include = True
                elif include_mlp and "mlp" in name:
                    include = True
                elif include_embeddings and "embeddings" in name: # Include both patch and position embeddings
                    include = True
                elif include_q_proj and "self_attn" in name and "q_proj" in name:
                    include = True
                elif include_k_proj and "self_attn" in name and "k_proj" in name:
                    include = True
                elif include_v_proj and "self_attn" in name and "v_proj" in name:
                    include = True
                elif include_o_proj and "self_attn" in name and "out_proj" in name:
                    include = True
                elif not include_self_attn and not include_mlp and not include_embeddings and not include_q_proj and not include_k_proj and not include_v_proj and not include_o_proj and not layer_types:
                    include = True

                if include:
                    filtered_name = name.replace(".weight", "")
                    filtered_names.append(filtered_name)

        elif include_language and "language_model" in name:
            layer_match = False
            if layers == "all" or (isinstance(language_layers, list) and any(f".layers.{layer_index}." in name for layer_index in considered_language_layers)) or (language_layers is None and isinstance(layers, list) and any(f".layers.{layer_index}." in name for layer_index in considered_language_layers)):
                layer_match = True

            if layer_match:
                include = False
                if include_self_attn and "self_attn" in name and any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                    include = True
                elif include_mlp and "mlp" in name:
                    include = True
                elif include_embeddings and "embed_tokens" in name:
                    include = True
                elif include_q_proj and "self_attn" in name and "q_proj" in name:
                    include = True
                elif include_k_proj and "self_attn" in name and "k_proj" in name:
                    include = True
                elif include_v_proj and "self_attn" in name and "v_proj" in name:
                    include = True
                elif include_o_proj and "self_attn" in name and "o_proj" in name:
                    include = True
                elif not include_self_attn and not include_mlp and not include_embeddings and not include_q_proj and not include_k_proj and not include_v_proj and not include_o_proj and not layer_types:
                    include = True

                if include:
                    filtered_name = name.replace(".weight", "")
                    filtered_names.append(filtered_name)

        if include_projector and "multi_modal_projector" in name:
            filtered_name = name.replace(".weight", "")
            filtered_names.append(filtered_name)

    return filtered_names

QUESTION_TYPE_MAPPING = {
    'descriptive': 0,
    'explanatory': 1,
    'predictive': 2,
    'counterfactual': 3,
}
REV_QUESTION_TYPE_MAPPING = {v: k for k, v in QUESTION_TYPE_MAPPING.items()}

def make_clevrer_collate_fn(model, processor, model_config, data_config, dtype):
    def collate_fn(examples):
        prompts, labels_text, images, question_types_list = [], [], [], []

        for example in examples:
            if example['question_type'] != 'descriptive':
                text = "ALWAYS Start your answer with 'Ans:'. Select all that apply. " + example["question"]
            else:
                text = "ALWAYS Start your answer with 'Ans:'. Answer with one word or number only. " + example["question"]

            if model_config.get('token_compression') is not None:
                num_image_tokens_calc = (model_config.get('target_length', 128) * data_config.get('num_frames', 8)) / 1024
                image_tokens_count = int(num_image_tokens_calc)
            else:
                image_tokens_count = data_config.get('num_frames')

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
        ).to(dtype).to(model.device)

        tokens["question_types"] = question_types_list
        return tokens

    return collate_fn

class CLEVRERTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        original_question_types = inputs.pop("question_types", None)

        inputs = self._prepare_inputs(inputs)

        # print(inputs)
        # print(inputs.keys())
        # print(inputs['input_ids'].shape)


        # # print(inputs['token_type_ids'].shape)
        # print(inputs['labels'].shape)



        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        # breakpoint()

        input_len = inputs["input_ids"].shape[-1]
        # # generation = generated_tokens[0][input_len:]
        # generation = generated_tokens[:, input_len:]
        # decoded = self.processing_class.batch_decode(generation, skip_special_tokens=True)
        # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # print(decoded)
        # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        # breakpoint()

        preds_token_ids = generated_tokens[:, input_len:]
        
        labels_token_ids = inputs.get("labels", None)
        mask = (labels_token_ids == -100)
        labels_token_ids.masked_fill_(mask, 0)

        # labels_token_ids = labels_token_ids[:, -2]

        # print(generated_tokens.shape)
        # print(labels_token_ids.shape)

        # breakpoint()

        question_type_ids = torch.tensor(
            [QUESTION_TYPE_MAPPING.get(q_type, -1) for q_type in original_question_types],
            dtype=torch.long,
            device=labels_token_ids.device
        ).unsqueeze(-1)

        combined_label_ids = torch.cat((labels_token_ids, question_type_ids), dim=1)

        # print(preds_token_ids)

        # print(generated_tokens.shape)
        # print(labels_token_ids.shape)

        # print("All occurrences (row, col) and then aggregated counts:")
        # row_indices, col_indices = (generated_tokens == 0).nonzero(as_tuple=True)

        # # Use a dictionary to store counts per row
        # row_counts = {}
        # for r_idx in row_indices.tolist():
        #     row_counts[r_idx] = row_counts.get(r_idx, 0) + 1

        # for row_idx, count in row_counts.items():
        #     print(f"GEN Row {row_idx}: Appears {count} time(s)")

        # token_type_ids = inputs['token_type_ids']
        # # generated_tokens = generated_tokens * token_type_ids
        # labels_token_ids = labels_token_ids * token_type_ids

        # print(generated_tokens)
        # # print(labels_token_ids)
        # # print(torch.equal(generated_tokens, labels_token_ids))

        # print("All occurrences (row, col) and then aggregated counts:")
        # row_indices, col_indices = (labels_token_ids == -100).nonzero(as_tuple=True)

        # # Use a dictionary to store counts per row
        # row_counts = {}
        # for r_idx in row_indices.tolist():
        #     row_counts[r_idx] = row_counts.get(r_idx, 0) + 1

        # for row_idx, count in row_counts.items():
        #     print(f"LABELS Row {row_idx}: Appears {count} time(s)")

        # print(generated_tokens.shape)

        # output = self.processing_class.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        # print(output)

        # output = self.processing_class.batch_decode(generated_tokens, skip_special_tokens=True)
        # print(output)

        # output = self.processing_class.batch_decode(labels_token_ids, skip_special_tokens=True)
        # print(output)

        return (None, preds_token_ids, combined_label_ids)

# def compute_accuracy(eval_preds):
    
#     # print("AAAAAAAAAAAAAAAAAAA")
#     # print("AAAAAAAAAAAAAAAAAAA")
#     # print("AAAAAAAAAAAAAAAAAAA")
#     # print("AAAAAAAAAAAAAAAAAAA")
#     # print("AAAAAAAAAAAAAAAAAAA")
#     # print("AAAAAAAAAAAAAAAAAAA")
#     # print("AAAAAAAAAAAAAAAAAAA")

#     preds_token_ids, combined_label_ids_tensor = eval_preds

#     if isinstance(combined_label_ids_tensor, np.ndarray):
#         combined_label_ids_tensor = torch.from_numpy(combined_label_ids_tensor)

#     question_type_ids = combined_label_ids_tensor[:, -1].tolist()
#     labels_token_ids = combined_label_ids_tensor[:, :-1]

#     # print(question_type_ids)

#     question_types = [REV_QUESTION_TYPE_MAPPING.get(q_id, "unknown") for q_id in question_type_ids]

#     correct_by_type = defaultdict(int)
#     total_by_type = defaultdict(int)

#     overall_correct = 0
#     overall_total = 0

#     print("\n")
#     print(preds_token_ids)
#     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
#     print(combined_label_ids_tensor)

#     breakpoint()
    
#     for i in range(len(preds_token_ids)):
#         row_a = [x.item() for x in preds_token_ids[i] if x.item() not in (0, 1, -100)]
#         row_b = [x.item() for x in labels_token_ids[i] if x.item() not in (0, 1, -100)]
#         # print(row_a)
#         # print(row_b)
#         is_correct = int(sorted(row_a) == sorted(row_b))

#         q_type = question_types[i]
#         # is_correct = is_correct_tensor[i].item()

#         total_by_type[q_type] += 1
#         if is_correct:
#             correct_by_type[q_type] += 1

#         overall_total += 1
#         if is_correct:
#             overall_correct += 1

#     metrics = {}
#     overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
#     metrics["overall_acc"] = overall_accuracy

#     for q_type in sorted(total_by_type.keys()):
#         if total_by_type[q_type] > 0:
#             type_accuracy = correct_by_type[q_type] / total_by_type[q_type]
#             metrics[f"{q_type}_acc"] = type_accuracy
#         else:
#             metrics[f"{q_type}_acc"] = 0.0

#     # print(metrics)

#     return metrics

def compute_accuracy(eval_preds):
    pred_tokens, label_tokens = eval_preds

    total_correct = 0
    total_samples = 0

    for i in range(len(pred_tokens)):
        pred_row = [x.item() for x in pred_tokens[i] if x.item() not in (0, 1, 2, 3, -100, 257152)] #special tokens ignored
        label_row = [x.item() for x in label_tokens[i] if x.item() not in (0, 1, 2, 3, -100, 257152)]

        is_correct = sorted(pred_row) == sorted(label_row)

        total_samples += 1
        if is_correct:
            total_correct += 1

    metrics = {}
    metrics["accuracy"] = total_correct/total_samples

    return metrics