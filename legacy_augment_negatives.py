import os, cv2, random, glob, numpy as np

def generate_synthetic_hard_negatives():
    print("[*] Legacy script: Generating synthetic red/orange hard negatives...")
    source_dir = os.path.join("data", "train", "non_fire")
    images = glob.glob(os.path.join(source_dir, "*.jpg"))
    if not images: return
        
    for i, img_path in enumerate(random.choices(images, k=100)):
        img = cv2.imread(img_path)
        if img is None: continue
        b, g, r = cv2.split(img)
        r_boost = np.clip(r.astype(np.uint16) + random.randint(60, 110), 0, 255).astype(np.uint8)
        b_dim = np.clip(b.astype(np.uint16) * 0.7, 0, 255).astype(np.uint8)
        g_dim = np.clip(g.astype(np.uint16) * 0.85, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(source_dir, f"synthetic_red_{i}.jpg"), cv2.merge((b_dim, g_dim, r_boost)))
    print("[+] Synthetic negatives created to combat color bias.")

if __name__ == "__main__":
    generate_synthetic_hard_negatives()