import sys
import os
import cv2
import torch
from ultralytics import YOLO

def main():
    if len(sys.argv) < 2:
        print("[-] Error: Please provide an image filename!")
        print("[*] Usage: python inference_det.py smoke2.jpg")
        return

    test_image_path = sys.argv[1]
    
    # Your clean weights
    weights_path = "runs/detect/runs/detect/train_final/weights/best.pt"
    
    # --- CUSTOM CLASS RULES ---
    SMOKE_CONF_THRESH = 0.05  # Keep low to catch faint smoke
    FIRE_CONF_THRESH = 0.25   # Keep high to stop false-positive fire
    # --------------------------
    
    print(f"[*] Loading model from: {weights_path}")
    model = YOLO(weights_path)
    
    # 1. Run at the lowest threshold so we don't miss anything initially
    base_conf = min(SMOKE_CONF_THRESH, FIRE_CONF_THRESH)
    results = model(test_image_path, conf=base_conf)[0]
    
    # 2. Get class IDs safely directly from the model
    smoke_id = [k for k, v in model.names.items() if v == 'smoke'][0]
    fire_id = [k for k, v in model.names.items() if v == 'fire'][0]

    # 3. Filter the boxes according to our custom rules
    valid_boxes = []
    for box in results.boxes.data:
        cls_id = int(box[5])
        conf = float(box[4])
        
        if cls_id == smoke_id and conf >= SMOKE_CONF_THRESH:
            valid_boxes.append(box)
        elif cls_id == fire_id and conf >= FIRE_CONF_THRESH:
            valid_boxes.append(box)
            
    # 4. Overwrite YOLO's results with only the boxes that passed our rules
    if valid_boxes:
        results.boxes.data = torch.stack(valid_boxes)
    else:
        results.boxes.data = torch.empty((0, 6), device=results.boxes.data.device)
        
    # 5. Draw and save the cleaned up image
    annotated_img = results.plot()
    
    os.makedirs("output", exist_ok=True)
    output_filename = f"detected_{os.path.basename(test_image_path)}"
    output_path = os.path.join("output", output_filename)
    
    cv2.imwrite(output_path, annotated_img)
    
    print(f"[+] Success! Custom thresholds applied. Check: {output_path}")

if __name__ == "__main__":
    main()