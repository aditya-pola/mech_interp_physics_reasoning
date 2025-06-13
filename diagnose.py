import os
import sys

def diagnose_issues():
    """Diagnose common issues with the evaluation setup."""
    
    print("=" * 60)
    print("MECH_INTERP_PHYSICS_REASONING DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # Check config files
    config_files = ['base_eval_config.yaml', 'config.yaml']
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"\n✓ Found {config_file}")
            if config_file == 'base_eval_config.yaml':
                with open(config_file, 'r') as f:
                    content = f.read()
                    if 'test_size: 2' in content:
                        print("  ⚠️  WARNING: test_size is set to 2 (too small!)")
                    elif 'test_size:' in content:
                        import re
                        match = re.search(r'test_size:\s*(\d+)', content)
                        if match:
                            test_size = int(match.group(1))
                            print(f"  ✓ test_size is set to {test_size}")
                            if test_size < 10:
                                print(f"    ⚠️  WARNING: test_size={test_size} might be too small for meaningful results")
        else:
            print(f"\n✗ Missing {config_file}")
    
    # Check data directories
    data_dirs = ['test_frames', 'train_frames', 'miscellaneous']
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"\n✓ Found {data_dir}/")
            if data_dir == 'test_frames':
                # Count video folders
                video_count = sum(1 for item in os.listdir(data_dir) 
                                if os.path.isdir(os.path.join(data_dir, item)) 
                                and item.startswith('video'))
                print(f"  Found {video_count} video folders")
            elif data_dir == 'miscellaneous':
                # Check for annotation files
                for ann_file in ['validation.json', 'train.json']:
                    ann_path = os.path.join(data_dir, ann_file)
                    if os.path.exists(ann_path):
                        print(f"  ✓ Found {ann_file}")
                    else:
                        print(f"  ✗ Missing {ann_file}")
        else:
            print(f"\n✗ Missing {data_dir}/")
    
    # Check Python environment
    print("\n\nPython Environment:")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check required packages
    required_packages = ['torch', 'transformers', 'peft', 'PIL', 'yaml', 'numpy']
    print("\nRequired packages:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (NOT INSTALLED)")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    print("""
1. If test_size is less than 10:
   - Edit base_eval_config.yaml and set test_size to at least 100
   - Or use 0.2 for 20% of the dataset

2. If packages are missing:
   - Create a new virtual environment with uv:
     uv venv .venv_new
   - Activate it and install packages:
     .venv_new\\Scripts\\activate (Windows)
     uv pip install torch torchvision transformers peft accelerate
     uv pip install Pillow numpy pyyaml wandb pytz

3. To run evaluation with base model:
   python scripts/eval.py --base

4. To run evaluation with a checkpoint:
   python scripts/eval.py path/to/checkpoint/folder

5. Common issues:
   - Empty predictions: Model might need more training or different prompting
   - CUDA errors: Set CUDA_VISIBLE_DEVICES or use cpu
   - OOM errors: Reduce eval_batch_size in config
""")

if __name__ == "__main__":
    diagnose_issues()
