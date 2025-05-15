# import os
# import json
# from PIL import Image
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# import string

# class ClevrerDataset(Dataset):
#     def __init__(self, frames_root, json_path, transform=None):
#         """
#         Args:
#             frames_root (string): Path to 'frame_captures' directory.
#             json_path (string): Path to 'train.json'.
#             transform (callable, optional): Optional transform to be applied on a frame.
#         """
#         self.frames_root = frames_root
#         self.transform = transform

#         # Load CLEVRER question/answer annotations
#         with open(json_path, 'r') as f:
#             self.annotations = json.load(f)

#         # Index annotations by video filename for quick access
#         self.annotations_by_video = {
#             entry['video_filename']: entry for entry in self.annotations
#         }

#         # Gather all video paths that have the 8 frames
#         self.video_dirs = []
#         self.video_filenames = []
#         for dirpath, _, filenames in os.walk(frames_root):
#             if all(f"frame_{i}.jpg" in filenames for i in range(8)):
#                 video_folder_name = os.path.basename(dirpath)
#                 video_filename = f"{video_folder_name}.mp4"
#                 if video_filename in self.annotations_by_video:
#                     self.video_dirs.append(dirpath)
#                     self.video_filenames.append(video_filename)


#     def __len__(self):
#         return len(self.video_dirs)

#     def __getitem__(self, idx):
#         video_dir = self.video_dirs[idx]
#         video_filename = self.video_filenames[idx]

#         # Load 8 frames
#         frames = []
#         for i in range(8):
#             frame_path = os.path.join(video_dir, f"frame_{i}.jpg")
#             image = Image.open(frame_path).convert("RGB")
#             if self.transform:
#                 image = self.transform(image)
#             frames.append(image)

#         # Get question/answer info
#         qa_entry = self.annotations_by_video[video_filename]
#         questions_by_type = {}

#         for q in qa_entry['questions']:
#             q_type = q['question_type']
#             if q_type not in questions_by_type:
#                 questions_by_type[q_type] = []

#             # Handle MCQ-type questions (with choices)
#             if q_type in ['counterfactual', 'explanatory', 'predictive']:
#                 base_question = q['question']

#                 choices = q.get('choices', [])

#                 lettered_choices = [f"{letter}) {c['choice']}" for letter, c in zip(string.ascii_lowercase, choices)]
#                 full_question = base_question + " " + " ".join(lettered_choices)

#                 correct_letters = [
#                     string.ascii_lowercase[i]
#                     for i, c in enumerate(choices)
#                     if c.get('answer', '') == 'correct'
#                 ]
#                 answer = " ".join(correct_letters)

#                 questions_by_type[q_type].append({
#                     'question_id': q['question_id'],
#                     'question': full_question,
#                     'answer': answer
#                 })

#             # Handle other question types (e.g., descriptive)
#             else:
#                 questions_by_type[q_type].append({
#                     'question_id': q['question_id'],
#                     'question': q['question'],
#                     'answer': q.get('answer', None)  # safe fallback
#                 })

#         sample = {
#             'frames': frames,
#             'questions': questions_by_type,
#             'video_filename': video_filename
#         }

#         return sample


import os
import json
from PIL import Image
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
import string
import random
from collections import defaultdict

NUM_FRAMES = 1

class ClevrerDataset(Dataset):
    def __init__(self, frames_root, json_path, transform=None, question_type='all'):
        self.frames_root = frames_root
        self.transform = transform
        self.question_type = question_type  # NEW: can be 'all', 'descriptive', 'counterfactual', etc.

        # Load CLEVRER annotations
        with open(json_path, 'r') as f:
            annotations = json.load(f)

        # Map video filename to question list
        self.annotations_by_video = {
            entry['video_filename']: entry for entry in annotations
        }

        # List of (video_dir, video_filename, question) tuples
        self.samples = []

        for dirpath, _, filenames in os.walk(frames_root):
            if all(f"frame_{i}.jpg" in filenames for i in range(NUM_FRAMES)):
                video_folder = os.path.basename(dirpath)
                video_filename = f"{video_folder}.mp4"

                if video_filename in self.annotations_by_video:
                    qa_entry = self.annotations_by_video[video_filename]
                    for q in qa_entry['questions']:
                        q_type = q['question_type']
                        if self.question_type != 'all' and q_type != self.question_type:
                            continue  # Skip irrelevant question types

                        # Handle MCQ-style questions
                        if q_type in ['counterfactual', 'explanatory', 'predictive']:
                            base_question = q['question']
                            choices = q.get('choices', [])
                            lettered_choices = [
                                f"{letter}) {c['choice']}" 
                                for letter, c in zip(string.ascii_lowercase, choices)
                            ]
                            full_question = base_question + " " + " ".join(lettered_choices)
                            correct_letters = [
                                string.ascii_lowercase[i]
                                for i, c in enumerate(choices)
                                if c.get('answer', '') == 'correct'
                            ]
                            answer = " ".join(correct_letters)
                        else:
                            full_question = q['question']
                            answer = q.get('answer', None)

                        self.samples.append((dirpath, video_filename, {
                            'question_id': q['question_id'],
                            'question_type': q_type,
                            'question': full_question,
                            'answer': answer
                        }))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, video_filename, question = self.samples[idx]

        # Load 8 frames
        frames = []
        for i in range(NUM_FRAMES):
            frame_path = os.path.join(video_dir, f"frame_{i}.jpg")
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        return {
            'frames': frames,
            'question_id': question['question_id'],
            'question_type': question['question_type'],
            'question': question['question'],
            'answer': question['answer'],
            'video_filename': video_filename
        }
    
    def train_test_split(self, test_size=0.2, random_seed=42, cache_path="split_indices.json"):
        """
        Performs a stratified train/test split based on question_type and saves/loads split indices.
        Returns:
            train_dataset (Subset), test_dataset (Subset)
        """
        # If cache exists, load and return split
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                saved = json.load(f)
            if saved.get("num_samples") == len(self.samples):
                return Subset(self, saved["train_indices"]), Subset(self, saved["test_indices"])

        # Create buckets per question_type
        buckets = defaultdict(list)
        for idx, (_, _, q) in enumerate(self.samples):
            buckets[q["question_type"]].append(idx)

        train_indices, test_indices = [], []

        rng = random.Random(random_seed)

        for qtype, idxs in buckets.items():
            rng.shuffle(idxs)
            split = int(test_size * len(idxs)) if isinstance(test_size, float) else int(test_size)
            test_indices.extend(idxs[:split])
            train_indices.extend(idxs[split:])

        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        # Save for reproducibility
        with open(cache_path, "w") as f:
            json.dump({
                "train_indices": train_indices,
                "test_indices": test_indices,
                "num_samples": len(self.samples)
            }, f, indent=2)

        return Subset(self, train_indices), Subset(self, test_indices)






if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ClevrerDataset(
        frames_root='frame_captures',
        json_path='train.json',
        transform=transform
    )

    sample = dataset[0]

    print(f"Frames: {len(sample['frames'])}")
    print(f"Video: {sample['video_filename']}")

    # print(sample.keys())
    # exit(0)

    print(sample)
    exit(0)

    for qtype, qlist in sample['question'].items():
        print(f"\n{qtype.upper()} QUESTIONS:")
        for q in qlist:
            print(f"Q: {q['question']}")
            print(f"A: {q['answer']}")
