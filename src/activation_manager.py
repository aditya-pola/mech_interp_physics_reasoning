from transformers import AutoProcessor, PreTrainedModel
import torch
from typing import List, Dict, Any, Callable, Tuple
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from src.torch_trace_utils import TraceDict

class ActivationReplacementExperiment:
    """
    Manages the extraction, storage, replacement, and evaluation of
    intermediate activations for custom datasets.
    """
    def __init__(self, model: PreTrainedModel, processor: AutoProcessor, device: str = "cuda:0",
                 activations_dir: str = "activations"):
        """
        Initializes the ActivationReplacementExperiment.

        Args:
            model: The pre-trained Transformer model.
            processor: The corresponding tokenizer/processor.
            device: The device to run computations on.
            activations_dir: Directory to store extracted activations.
        """
        self.model = model.to(device).eval()
        self.processor = processor
        self.device = device
        self.activations_dir = activations_dir
        os.makedirs(self.activations_dir, exist_ok=True)
        self.real_activations: Dict[str, Dict[int, torch.Tensor]] = {}  # {layer_name: {sample_index: activation}}
        self.corrupt_activations: Dict[str, Dict[int, torch.Tensor]] = {} # {layer_name: {sample_index: activation}}
            
    def _extract_activations(self, dataloader: DataLoader, layer_names: List[str], store_as: str) -> None:
            """
            Extracts and stores intermediate activations for the specified layers
            from a given dataloader, processing image and text inputs.

            Args:
                dataloader: DataLoader for the dataset, yielding batches with 'frames' (list of 8 images) and 'question'.
                layer_names: List of module names to extract activations for.
                store_as: Key to store the activations in ('real' or 'corrupt').
            """
            activations_dict = self.real_activations if store_as == 'real' else self.corrupt_activations
            for batch_idx, batch in enumerate(dataloader):
                frames_list = batch['frames']
                questions = batch['question']

                # print(questions)
                questions_updated = []

                for question in questions:
                    image_tokens = "<image> " * 8
                    prompt = image_tokens.strip() + " " + question + " en"
                    questions_updated.append(prompt) 
                
                # print(len(list(frames_list)))
                # print(frames_list)
                # print("************************")
                # print(list(frames_list))
                # print(questions)
                # print(questions_updated)

                frame_set = []
                frame_set.append(frames_list)

                # Process the batch using the PaliGemma processor
                # processed_inputs = self.processor(images=[frames_list], text=questions, return_tensors="pt").to(self.device)
                processed_inputs = self.processor(images=frame_set, text=questions_updated, return_tensors="pt").to(self.device)

                print(processed_inputs.keys())
                print(processed_inputs['input_ids'].shape)

                with torch.inference_mode():
                    with TraceDict(self.model, layers=layer_names, retain_output=True, detach=True) as tr:
                        # Perform a forward pass with the processed inputs
                        _ = self.model(**processed_inputs)
                        for layer_name in layer_names:
                            if layer_name in tr:
                                if layer_name not in activations_dict:
                                    activations_dict[layer_name] = {}
                                # batch_size = processed_inputs['pixel_values'].shape[0] # Get batch size from processed inputs

                                # print(batch_size)
                                # exit(0)


                                activations_dict[layer_name] = tr[layer_name].output.cpu()

                                # for sample_idx_in_batch in range(batch_size):
                                #     global_sample_idx = batch_idx * dataloader.batch_size + sample_idx_in_batch
                                #     activations_dict[layer_name][global_sample_idx] = tr[layer_name].output[sample_idx_in_batch].cpu()
                            else:
                                print(f"Warning: Layer '{layer_name}' not found during extraction.")
            print(f"Extracted and stored '{store_as}' activations for {len(dataloader.dataset)} samples.")

    def load_activations(self, store_as: str) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Loads previously stored activations from disk.

        Args:
            store_as: Key indicating which set of activations to load ('real' or 'corrupt').

        Returns:
            A dictionary containing the loaded activations, or an empty dict if not found.
        """
        filename = os.path.join(self.activations_dir, f"{store_as}_activations.pkl")
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Warning: Activations file for '{store_as}' not found at {filename}.")
            return {}

    def save_activations(self, store_as: str) -> None:
        """
        Saves the currently stored activations to disk.

        Args:
            store_as: Key indicating which set of activations to save ('real' or 'corrupt').
        """
        activations_to_save = self.real_activations if store_as == 'real' else self.corrupt_activations
        filename = os.path.join(self.activations_dir, f"{store_as}_activations.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(activations_to_save, f)
        print(f"Saved '{store_as}' activations to {filename}.")

    def replace_and_evaluate(self, eval_dataloader: DataLoader, replace_config: Dict[str, Tuple[int, ...]], metric_fn: Callable) -> float:
        """
        Performs a forward pass on the evaluation dataset, replacing specified
        layers' activations with the stored 'real' activations.

        Args:
            eval_dataloader: DataLoader for the evaluation dataset.
            replace_config: A dictionary specifying which layers to replace and
                            the index of the 'real' sample to use for replacement.
                            Example: {'vision_tower.layer.5.attn': (real_sample_index,),
                                      'language_model.layer.10.mlp': (real_sample_index,)}
            metric_fn: A function that takes the model's output and the ground truth
                       and returns the evaluation metric.

        Returns:
            The evaluation metric on the eval_dataloader after the activation replacement.
        """
        self.model.eval()
        total_metric = 0
        num_batches = len(eval_dataloader)

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                ground_truth = self._get_ground_truth(batch) # Implement this based on your task

                def edit_output_fn(output: torch.Tensor, name: str) -> torch.Tensor:
                    if name in replace_config:
                        real_sample_index = replace_config[name][0]
                        if name in self.real_activations and real_sample_index in self.real_activations[name]:
                            # Assuming batch size 1 for simplicity in replacement for now
                            return self.real_activations[name][real_sample_index].to(output.device).unsqueeze(0)
                        else:
                            print(f"Warning: Real activation for layer '{name}' and sample {real_sample_index} not found.")
                    return output

                layers_to_trace = list(replace_config.keys())
                with TraceDict(self.model, layers=layers_to_trace, retain_output=True, detach=False,
                               edit_output={layer: edit_output_fn for layer in layers_to_trace}, stop=True) as tr:
                    outputs = self.model(**batch)
                    total_metric += metric_fn(outputs, ground_truth).item()

        return total_metric / num_batches

    def run_sequential_replacement(self, eval_dataset: Dataset, corrupt_dataloader: DataLoader,
                                   layer_names_to_replace: List[str], real_sample_indices: List[int],
                                   corrupt_sample_index: int, metric_fn: Callable,
                                   replacement_unit: str = 'layer', replacement_window: int = 1) -> List[float]:
        """
        Sequentially replaces activations of layers or (to be implemented) attention heads
        from a 'corrupt' forward pass with 'real' activations and evaluates the
        performance on the evaluation dataset after each step.

        Args:
            eval_dataset: The dataset to evaluate on.
            corrupt_dataloader: DataLoader for the 'corrupt' data to get initial activations from.
            layer_names_to_replace: The list of layer names whose activations will be replaced.
            real_sample_indices: The indices of the 'real' samples to use for replacement.
            corrupt_sample_index: The index of the 'corrupt' sample whose activations are being replaced.
            metric_fn: The evaluation metric function.
            replacement_unit: Whether to replace by 'layer' (currently supported) or 'head'.
            replacement_window: The number of layers or heads to replace at each step.

        Returns:
            A list of evaluation metrics after each sequential replacement step.
        """
        evaluation_metrics = []
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False) # Process one eval sample at a time for now

        # Get the 'corrupt' input
        corrupt_batch = next(iter(corrupt_dataloader))
        corrupt_batch = {k: v.to(self.device) for k, v in corrupt_batch.items()}

        with torch.no_grad():
            with TraceDict(self.model, layers=layer_names_to_replace, retain_output=True, detach=False) as corrupt_tr:
                _ = self.model(**corrupt_batch)
                corrupt_layer_outputs = {name: corrupt_tr[name].output[0].cpu() for name in layer_names_to_replace if name in corrupt_tr}

            num_layers_to_replace = len(layer_names_to_replace)
            for i in range(0, num_layers_to_replace, replacement_window):
                layers_to_replace_now = layer_names_to_replace[i:i + replacement_window]
                current_metric = 0
                num_eval_samples = len(eval_dataset)

                for eval_sample_idx in range(num_eval_samples):
                    eval_sample = eval_dataset[eval_sample_idx]
                    eval_batch = self.processor({k: v for k, v in eval_sample.items() if k in ['text', 'images']}, return_tensors="pt").to(self.device)
                    ground_truth = self._get_ground_truth(eval_batch)