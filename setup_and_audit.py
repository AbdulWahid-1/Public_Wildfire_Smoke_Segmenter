import os
import sys
import glob
import yaml
import matplotlib.pyplot as plt

def main():
    print("[*] STEP 1: Verifying Environment & Drivers...")
    try:
        import torch
        import ultralytics
        import cv2
        print("  [+] ALL REQUIRED LIBRARIES FOUND.")
        if torch.cuda.is_available():
            print(f"  [+] GPU ACCELERATION ACTIVE: {torch.cuda.get_device_name(0)}")
        else:
            print("  [-] WARNING: CUDA not detected. Training will fall back to CPU and be slow.")
    except ImportError as e:
        print(f"  [-] MISSING DRIVER/PACKAGE: {e}")
        sys.exit(1)

    print("\n[*] STEP 2: Auto-Detecting Dataset Configuration...")
    dataset_root = None
    
    # Smart search to find exactly where the 'train' and 'val' folders actually are
    for root_dir, dirs, files in os.walk("datasets"):
        if "train" in dirs and "val" in dirs:
            dataset_root = os.path.abspath(root_dir)
            break
            
    if not dataset_root:
        print("  [-] Error: Could not find 'train' and 'val' folders anywhere inside the 'datasets' directory.")
        print("  [*] Are you sure you extracted the Kaggle .zip file? Check your folders.")
        sys.exit(1)
        
    print(f"  [+] Found dataset root actively at: {dataset_root}")
    
    # Save the YAML directly to datasets/fire_detect/ so YOLO can find it easily
    yaml_path = os.path.join("datasets", "fire_detect", "data.yaml")
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    
    # Auto-detect if images are in a subfolder (train/images) or directly in the folder (train/)
    train_path = 'train/images' if os.path.exists(os.path.join(dataset_root, 'train', 'images')) else 'train'
    val_path = 'val/images' if os.path.exists(os.path.join(dataset_root, 'val', 'images')) else 'val'
    test_path = 'test/images' if os.path.exists(os.path.join(dataset_root, 'test', 'images')) else ('test' if os.path.exists(os.path.join(dataset_root, 'test')) else '')
    
    # Dynamically building the YOLO config file
    yaml_content = {
        'path': dataset_root,
        'train': train_path, 
        'val': val_path,
        'nc': 2,
        'names': ['fire', 'smoke']
    }
    
    if test_path:
        yaml_content['test'] = test_path
        
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"  [+] Successfully generated YOLO config at: {yaml_path}")
    
    print("\n[*] STEP 3: Auditing Dataset Structure...")
    os.makedirs("output", exist_ok=True)
    
    # Count images dynamically (.jpg or .png) based on where they actually are
    train_images = len(glob.glob(os.path.join(dataset_root, train_path, "*.jpg"))) + len(glob.glob(os.path.join(dataset_root, train_path, "*.png")))
    val_images = len(glob.glob(os.path.join(dataset_root, val_path, "*.jpg"))) + len(glob.glob(os.path.join(dataset_root, val_path, "*.png")))
    
    print(f"  [+] Found {train_images} Training Images.")
    print(f"  [+] Found {val_images} Validation Images.")
    
    # Generate data prep visualization for the README
    if train_images > 0:
        labels = ['Training Set', 'Validation Set']
        counts = [train_images, val_images]
        plt.figure(figsize=(8, 5))
        plt.bar(labels, counts, color=['#ff9999', '#66b3ff'])
        plt.title('Dataset Split Distribution (Object Detection)')
        plt.ylabel('Number of Images')
        
        chart_path = os.path.join("output", "dataset_audit.png")
        plt.savefig(chart_path)
        print(f"  [+] Dataset audit chart saved to {chart_path} for your README.")
        print("\n[+] SUCCESS: run: python train_det.py")
    else:
        print("  [-] WARNING: No images found in the auto-detected folders. Check the extraction.")

if __name__ == "__main__":
    main()