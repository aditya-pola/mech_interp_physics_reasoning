#!/usr/bin/env python
"""
Quick fix script for mech_interp_physics_reasoning evaluation issues.
This script:
1. Fixes the test_size in config
2. Provides commands to set up the environment
3. Shows how to properly run the evaluation
"""

import os
import yaml
import sys

def fix_config():
    """Fix the test_size in base_eval_config.yaml"""
    config_path = "base_eval_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found!")
        return False
    
    # Read config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check current test_size
    old_test_size = config['data_config']['test_size']
    print(f"Current test_size: {old_test_size}")
    
    if old_test_size < 10:
        # Update test_size
        config['data_config']['test_size'] = 100  # or 0.2 for 20%
        
        # Write back
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Updated test_size to: {config['data_config']['test_size']}")
        return True
    else:
        print("✓ test_size is already adequate")
        return True

def main():
    print("MECH_INTERP_PHYSICS_REASONING - Quick Fix Tool")
    print("=" * 50)
    
    # Fix config
    print("\n1. Fixing configuration...")
    fix_config()
    
    # Show environment setup commands
    print("\n2. Environment Setup Commands:")
    print("   If you haven't already, run these commands:")
    print("   ```")
    print("   # Create new virtual environment")
    print("   uv venv .venv_new")
    print("   ")
    print("   # Activate it (Windows)")
    print("   .venv_new\\Scripts\\activate")
    print("   ")
    print("   # Install packages")
    print("   uv pip install torch torchvision transformers peft accelerate")
    print("   uv pip install Pillow numpy pyyaml wandb pytz")
    print("   ```")
    
    # Show evaluation commands
    print("\n3. To run evaluation:")
    print("   ```")
    print("   # For base model evaluation:")
    print("   python scripts/eval.py --base")
    print("   ")
    print("   # For checkpoint evaluation:")
    print("   python scripts/eval.py path/to/checkpoint")
    print("   ```")
    
    # Additional debugging tips
    print("\n4. Debugging Tips:")
    print("   - If getting CUDA errors, set: os.environ['CUDA_VISIBLE_DEVICES'] = '0'")
    print("   - If OOM, reduce eval_batch_size in config")
    print("   - Check test_frames/ has video folders")
    print("   - Check miscellaneous/validation.json exists")
    
    print("\n✓ Quick fix complete!")

if __name__ == "__main__":
    main()
