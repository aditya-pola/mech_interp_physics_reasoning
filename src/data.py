import os
import json
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import string
import random

class ClevrerDataset(Dataset):
    def __init__(self, frames_root, json_path, transform=None, question_type='all', NUM_FRAMES=8):
        self.frames_root = frames_root
        self.transform = transform
        self.question_type = question_type
        self.NUM_FRAMES = NUM_FRAMES

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotation file not found at: {json_path}")

        with open(json_path, 'r') as f:
            annotations = json.load(f)

        self.annotations_by_video = {
            entry['video_filename']: entry for entry in annotations
        }

        self.samples = []

        if not os.path.exists(frames_root):
            print(f"Warning: Frames root directory not found: {frames_root}. Dataset will be empty.")
            return

        for dirpath, _, filenames in os.walk(frames_root):
            is_video_folder = os.path.basename(dirpath).startswith("video")
            has_all_frames = all(f"frame_{i}.jpg" in filenames for i in range(self.NUM_FRAMES))

            if is_video_folder and has_all_frames:
                video_folder = os.path.basename(dirpath)
                video_filename = f"{video_folder}.mp4"

                if video_filename in self.annotations_by_video:
                    qa_entry = self.annotations_by_video[video_filename]
                    for q in qa_entry['questions']:
                        q_type = q['question_type']
                        if self.question_type != 'all' and q_type != self.question_type:
                            continue

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
                else:
                    # print(f"Warning: Video filename {video_filename} from folder {video_folder} not found in annotations.")
                    pass # Suppress warning for cases where frames exist but no annotations


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, video_filename, question = self.samples[idx]

        frames = []
        for i in range(self.NUM_FRAMES):
            frame_path = os.path.join(video_dir, f"frame_{i}.jpg")
            if os.path.exists(frame_path):
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

    def train_test_split(self, test_size=0.2, random_seed=42, cache_path="miscellaneous/split_indices.json"):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                saved = json.load(f)
            if saved.get("num_samples") == len(self.samples):
                # print(f"Loading split from cache: {cache_path}")
                return Subset(self, saved["train_indices"]), Subset(self, saved["test_indices"])
            else:
                print("Cache found but number of samples changed or cache is for a different dataset. Regenerating split.")
                # Clean up old cache if it's invalid to avoid confusion
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    print(f"Removed invalid cache file: {cache_path}")

        videos_in_dataset = set(s[1] for s in self.samples)
        all_video_filenames = sorted(list(videos_in_dataset))

        if not all_video_filenames:
            print("No videos found in the dataset with associated questions. Cannot perform split.")
            return Subset(self, []), Subset(self, []) # Return empty subsets

        rng = random.Random(random_seed)
        rng.shuffle(all_video_filenames)

        num_test_videos = int(len(all_video_filenames) * test_size)
        if num_test_videos == 0 and len(all_video_filenames) > 0 and test_size > 0:
            # Ensure at least one video in test set if possible
            num_test_videos = 1
            print(f"Warning: test_size {test_size} resulted in 0 test videos. Setting to 1 test video.")
        elif num_test_videos == len(all_video_filenames) and test_size < 1.0:
             # Ensure at least one video in train set if possible
            num_test_videos = len(all_video_filenames) - 1
            print(f"Warning: test_size {test_size} resulted in all videos in test set. Setting to {len(all_video_filenames) - 1} test videos.")

        test_video_filenames = set(all_video_filenames[:num_test_videos])
        train_video_filenames = set(all_video_filenames[num_test_videos:])

        print(f"Splitting {len(all_video_filenames)} unique videos: {len(train_video_filenames)} for train, {len(test_video_filenames)} for test.")

        train_indices, test_indices = [], []
        for idx, (_, video_filename, _) in enumerate(self.samples):
            if video_filename in train_video_filenames:
                train_indices.append(idx)
            elif video_filename in test_video_filenames:
                test_indices.append(idx)
            # If a video is neither in train nor test, it will be skipped from the split (shouldn't happen with proper splitting logic)

        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        with open(cache_path, "w") as f:
            json.dump({
                "train_indices": train_indices,
                "test_indices": test_indices,
                "num_samples": len(self.samples)
            }, f, indent=2)
        print(f"Split generated and saved to cache: {cache_path}")

        return Subset(self, train_indices), Subset(self, test_indices)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    frames_root_path = 'frame_captures'
    json_annotations_path = 'miscellaneous/train.json'
    split_cache_path = 'miscellaneous/split_indices.json'

    if not os.path.exists(frames_root_path):
        print(f"Error: Frame captures directory not found at '{frames_root_path}'. Please create it and populate with frames.")
        exit()
    if not os.path.exists(json_annotations_path):
        print(f"Error: Annotation file not found at '{json_annotations_path}'. Please provide the correct path.")
        exit()
    if os.path.exists(split_cache_path):
        print(f"Removing existing cache file: {split_cache_path} for a fresh test.")
        os.remove(split_cache_path)


    print("\n--- Initializing Dataset ---")
    dataset = ClevrerDataset(
        frames_root=frames_root_path,
        json_path=json_annotations_path,
        transform=transform
    )

    print(f"Total samples in dataset: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty. Please ensure 'frame_captures' contains video frames and 'miscellaneous/train.json' is correctly formatted with matching video filenames.")
        exit()

    print("\n--- Testing Sample Retrieval ---")
    try:
        sample = dataset[0] # Get the first sample
        print(f"Successfully retrieved sample 0:")
        print(f"  Number of frames: {len(sample['frames'])}")
        print(f"  Video Filename: {sample['video_filename']}")
        print(f"  Question ID: {sample['question_id']}")
        print(f"  Question Type: {sample['question_type']}")
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer']}")
    except IndexError:
        print("Could not retrieve sample 0. Dataset might be misconfigured or empty.")
    except Exception as e:
        print(f"An error occurred while retrieving sample 0: {e}")


    print("\n--- Verifying Train/Test Split ---")
    test_size = 0.2
    try:
        train_subset, test_subset = dataset.train_test_split(
            test_size=test_size,
            random_seed=42,
            cache_path=split_cache_path
        )

        print(f"Train subset size: {len(train_subset)}")
        print(f"Test subset size: {len(test_subset)}")

        # Check 1: Disjoint Video Filenames
        train_videos = set()
        for idx in train_subset.indices:
            # dataset.samples[idx][1] gets the video_filename from the (dirpath, video_filename, question_dict) tuple
            train_videos.add(dataset.samples[idx][1])

        test_videos = set()
        for idx in test_subset.indices:
            test_videos.add(dataset.samples[idx][1])

        common_videos = train_videos.intersection(test_videos)

        print(f"Number of unique videos in train set: {len(train_videos)}")
        print(f"Number of unique videos in test set: {len(test_videos)}")
        print(f"Number of common videos between train and test sets: {len(common_videos)}")

        if len(common_videos) == 0:
            print("VERIFICATION SUCCESS: Train and Test sets have disjoint video filenames!")
        else:
            print(f"VERIFICATION FAILED: Found {len(common_videos)} common videos in train and test sets: {list(common_videos)}")
            # Optionally print some common videos for debugging
            # print("Example common videos:", list(common_videos)[:5])


        # Check 2: Total number of samples matches
        total_split_samples = len(train_subset) + len(test_subset)
        if total_split_samples == len(dataset):
            print(f"VERIFICATION SUCCESS: Total samples in splits ({total_split_samples}) match total dataset samples ({len(dataset)}).")
        else:
            print(f"VERIFICATION FAILED: Total samples in splits ({total_split_samples}) do NOT match total dataset samples ({len(dataset)}).")

        # Check 3: Test size proportion (approximate)
        # This will be approximate because we are splitting by video, not individual samples.
        # The number of questions per video can vary.
        expected_test_samples_min = (len(dataset) / len(dataset.samples)) * (len(dataset.samples) * test_size - dataset.NUM_FRAMES * dataset.NUM_FRAMES / 2) # Rough lower bound
        expected_test_samples = len(dataset) * test_size
        print(f"Expected test samples (approx): {expected_test_samples:.2f}")
        print(f"Actual test samples: {len(test_subset)}")
        # A simple check: test size is within 50% of expected, or within a small absolute difference.
        # This tolerance might need adjustment based on your data distribution.
        if abs(len(test_subset) - expected_test_samples) < (len(dataset) * 0.1) or len(dataset) < 10: # Allow 10% deviation, or if dataset is small
            print("VERIFICATION SUCCESS: Test subset size is approximately as expected.")
        else:
            print("VERIFICATION WARNING: Test subset size deviates significantly from expected proportion.")


        print("\n--- Testing Cache Loading ---")
        # Ensure the cache file exists from the previous split call
        if os.path.exists(split_cache_path):
            train_subset_cached, test_subset_cached = dataset.train_test_split(
                test_size=test_size,
                random_seed=42,
                cache_path=split_cache_path
            )
            if len(train_subset_cached) == len(train_subset) and len(test_subset_cached) == len(test_subset):
                print("VERIFICATION SUCCESS: Cache loading produced identical split sizes.")
            else:
                print("VERIFICATION FAILED: Cache loading produced different split sizes.")
        else:
            print("Cache file was not created, cannot test cache loading.")


    except Exception as e:
        print(f"\nAn error occurred during train/test split verification: {e}")