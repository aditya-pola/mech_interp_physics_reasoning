import wandb
import os
from torchvision import transforms
from data import ClevrerDataset
from datetime import datetime
import pytz
import re

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

    train_ds, val_ds = dataset.train_test_split(test_size=data_config.get('test_size'))
    return train_ds, val_ds

def generate_run_name(config):
    """
    Generates a run name string based on the provided configuration.

    Args:
        config (dict): A dictionary containing 'model_train', 'data_config',
                       and 'lora' configurations.

    Returns:
        str: The dynamically generated run name.
    """
    now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))

    # Extract relevant config details
    full_model_name = config['model_train']['model']
    # Shorten model name to "PaliG" if it's the specific Google model, otherwise use the last part of the name
    model_name_short = "PaliG" if full_model_name == "google/paligemma2-3b-mix-448" else full_model_name.split('/')[-1]
    
    num_epochs = config['model_train']['num_epochs']
    batch_size = config['model_train']['batch_size']
    learning_rate = config['model_train']['learning_rate']
    image_size = config['data_config']['image_size']
    num_frames = config['data_config']['num_frames']

    token_compression = config['model_train']['token_compression']
    target_length = config['model_train']['target_length']

    lora_config = config['lora']
    lora_rank = lora_config['rank']
    lora_layers = lora_config['layers']
    lora_layer_types = lora_config.get('layer_types', [])
    lora_include_vision = lora_config['include_vision']
    lora_include_language = lora_config['include_language']
    lora_vision_layers = lora_config.get('vision_layers')
    lora_language_layers = lora_config.get('language_layers')

    # Succinct layer_types mapping
    layer_type_map = {
        "self_attn": "attn",
        "mlp": "mlp",
        "embeddings": "emb",
        "projector": "proj",
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
        "o_proj": "o"
    }
    succinct_layer_types = [layer_type_map.get(lt, lt) for lt in lora_layer_types]
    layer_types_str = "_".join(succinct_layer_types) if succinct_layer_types else "all_types"

    # Determine layers string for run name
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
        # Fallback if 'layers' is a list but vision/language specific layers aren't set
        layers_str = "selected_layers"

    # Determine inclusion mode
    inclusion_mode = ""
    if lora_include_vision and lora_include_language:
        inclusion_mode = "vision_and_language"
    elif lora_include_vision and not lora_include_language:
        inclusion_mode = "only_vision"
    elif not lora_include_vision and lora_include_language:
        inclusion_mode = "only_language"
    else:
        inclusion_mode = "no_vision_no_language"

    # Construct run name parts
    dynamic_run_name_parts = [
        now_ist.strftime("%m-%d_%H:%M"),
        model_name_short,
        f"frames={num_frames}",
        f"lora: r={lora_rank}",
        inclusion_mode,
        f"{layers_str}->{layer_types_str}",
        f"{token_compression}-compr_1024->{target_length}toks",
        f"e={num_epochs}",
        f"b={batch_size}",
        f"lr={learning_rate}",
        f"im_size={image_size}",
    ]

    # Join all parts with spaces
    dynamic_run_name = " ".join(dynamic_run_name_parts)
    return dynamic_run_name

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

def setup_wandb(project_name, run_name, key):
    try:
        wandb.login(key=key)
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