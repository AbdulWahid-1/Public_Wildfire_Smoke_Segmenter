import sys
try:
    import torch
    import ultralytics
    import cv2
    print("[+] ALL LIBRARIES FOUND.")
    print(f"[*] PyTorch Version: {torch.__version__}")
    print(f"[*] Ultralytics Version: {ultralytics.__version__}")
    
    if torch.cuda.is_available():
        print(f"[+] GPU ACCELERATION ACTIVE: {torch.cuda.get_device_name(0)}")
    else:
        print("[-] WARNING: CUDA not detected. Training will run on CPU and be very slow.")
except ImportError as e:
    print(f"[-] MISSING DRIVER/PACKAGE: {e}")
    sys.exit(1)