# from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
# import torch
# from torch.utils.data import DataLoader
# from data import ClevrerDataset
# import torchvision.transforms as transforms
# import numpy as np


# # --- Setup ---
# model_id = "google/paligemma2-3b-mix-448"
# device = torch.device("cuda:3")

# # Load model and processor
# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval().to(device)
# processor = PaliGemmaProcessor.from_pretrained(model_id)

# transform = transforms.Compose([
#     transforms.Resize((448, 448)),
#     transforms.ToTensor(),
# ])

# dataset = ClevrerDataset(
#     frames_root='frame_captures',
#     json_path='train.json',
#     transform=transform
# )

# # print(len(dataset))

# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# # --- Inference for a single sample ---
# data_iter = iter(dataloader)
# sample = next(data_iter)

# frames = sample['frames']  # List of 8 PIL images

# questions_by_type = sample['questions']
# video_name = sample['video_filename'][0]

# print(video_name)

# # Process each question type
# for qtype, questions in questions_by_type.items():
#     print(f"\n== {qtype.upper()} QUESTIONS ==")
#     for q in questions:
#         text_prompt = q['question'][0] + " en"

#         if qtype != "descriptive":
#             text_prompt = "Select all that apply. " + text_prompt

#         image_tokens = "<image><image><image><image><image><image><image><image>" 
#         text_prompt = image_tokens + " " + text_prompt
#         print(text_prompt)

#         # Process text + image together for each question
#         model_inputs = processor(text=text_prompt, images=frames, do_rescale=False, return_tensors="pt").to(device, torch.bfloat16)

#         # print(model_inputs.keys())
#         # print(model_inputs["input_ids"].shape)
#         # print(model_inputs["pixel_values"].shape)
#         # print(model_inputs["attention_mask"].shape)

#         input_len = model_inputs["input_ids"].shape[-1]

#         with torch.inference_mode():
#             output = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
#             print(output.shape)
#             output = output[0][input_len:]
#             decoded = processor.decode(output, skip_special_tokens=True)

#         print(f"Q: {text_prompt}")
#         print(f"A: {decoded}")



from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from data import ClevrerDataset
import torchvision.transforms as transforms
import numpy as np

# --- Setup ---
model_id = "google/paligemma2-3b-mix-448"
device = torch.device("cuda:2")

# Load model and processor
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval().to(device)
processor = PaliGemmaProcessor.from_pretrained(model_id)

# --- Transform (with NumPy conversion for processor compatibility) ---
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    # lambda x: x.mul(255).byte().permute(1, 2, 0).numpy()  # (H, W, C) uint8
])

# --- Custom collate function ---
def clevrer_collate_fn(batch):
    return {
        'frames': [sample['frames'] for sample in batch],
        'question': [sample['question'] for sample in batch],
        'question_id': [sample['question_id'] for sample in batch],
        'question_type': [sample['question_type'] for sample in batch],
        'answer': [sample['answer'] for sample in batch],
        'video_filename': [sample['video_filename'] for sample in batch],
    }

# --- Dataset and Dataloader ---
dataset = ClevrerDataset(
    frames_root='frame_captures',
    json_path='train.json',
    transform=transform,
    question_type="counterfactual"
)

# print(len(dataset))
# exit(0)

dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=clevrer_collate_fn)

# --- Inference loop ---
for batch in dataloader:
    all_prompts = []
    all_images = []

    for i in range(len(batch['question'])):
        q_text = batch['question'][i]
        qtype = batch['question_type'][i]

        if qtype != "descriptive":
            q_text = "Select all that apply. " + q_text

        image_tokens = "<image> " * 8
        prompt = image_tokens.strip() + " " + q_text + " en"
        all_prompts.append(prompt)
        all_images.append(batch['frames'][i])  # list of 8 numpy images

    # try:
        # Batch process text + image
    model_inputs = processor(
        text=all_prompts,
        images=all_images,
        return_tensors="pt",
        do_rescale=False,
        padding=True
    ).to(device, torch.bfloat16)

    input_len = model_inputs["input_ids"].shape[1]


    with torch.inference_mode():
        outputs = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)

    print("========================================")

    # Decode results
    for i, output in enumerate(outputs):
        decoded = processor.decode(output[input_len:], skip_special_tokens=True)
        print(f"Video: {batch['video_filename'][i]}")
        print(f"QID: {batch['question_id'][i]}")
        print(f"Q: {batch['question'][i]}")
        print(f"GT: {batch['answer'][i]}")
        print(f"Predicted: {decoded}")
        print()

    # except Exception as e:
    #     print("⚠️  Error during model.generate or decode:")
    #     print(f"Video filenames in batch: {batch['video_filename']}")
    #     print(f"Exception: {e}")
    #     print("Skipping this batch.\n")
    #     continue    