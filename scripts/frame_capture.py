import os
import cv2
import numpy as np
from PIL import Image

# Function to extract 8 evenly spaced frames from a video
def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))  # Convert OpenCV frame to PIL Image

    cap.release()
    return frames

# Root directory containing videos
input_root = "/data/ai24mtech02001/DATA/CLEVRER/video_validation"
output_base = os.path.join("/data/ai24mtech02001/projects/mech_interp_physics_reasoning/", "test_frames")

# Walk through all subfolders and files
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            print(f"Processing: {video_path}")

            # Extract relative path to preserve folder structure
            relative_path = os.path.relpath(video_path, input_root)
            relative_dir = os.path.dirname(relative_path)
            video_name = os.path.splitext(file)[0]

            # Create output directory: current_dir/frame_captures/relative_path/video_name
            output_dir = os.path.join(output_base, relative_dir, video_name)
            os.makedirs(output_dir, exist_ok=True)

            # Extract and save frames
            frames = extract_frames(video_path, num_frames=8)
            for i, frame in enumerate(frames):
                frame.save(os.path.join(output_dir, f"frame_{i}.jpg"))

print("All videos processed and frames saved in 'frame_captures'.")
