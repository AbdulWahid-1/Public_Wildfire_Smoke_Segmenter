from ultralytics import YOLO

def main():
    # Start fresh. Do NOT use the old, corrupted 'last.pt' or 'best.pt'
    model = YOLO("yolov8n.pt")
    
    print("[*] Starting clean training on the perfected dataset...")
    
    model.train(
        data="datasets/fire_detect/data.yaml",
        epochs=30, # Back to 30 for a full, proper learning cycle
        imgsz=640,
        batch=16,
        project="runs",
        name="detect/train_final",
        plots=True 
    )

if __name__ == "__main__":
    main()
    