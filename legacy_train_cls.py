from ultralytics import YOLO

def main():
    print("[!] Running Legacy Phase 2 Classification Training...")
    # This architecture was abandoned due to Global Color Bias (Minecraft house failure)
    model = YOLO("yolov8n-cls.pt")
    model.train(data="data", epochs=10, imgsz=224, project="runs", name="classify/train")

if __name__ == "__main__":
    main()