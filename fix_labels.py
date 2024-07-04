import glob

def normalize_dataset():
    print("[*] Auditing dataset labels...")
    label_files = glob.glob("datasets/fire_detect/fire_smoke/**/labels/*.txt", recursive=True)
    
    fixed_count = 0
    for file_path in label_files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        changed = False
        for line in lines:
            parts = line.strip().split()
            # If the label is Class 2, change it to Class 0 (Smoke)
            if len(parts) > 0 and parts[0] == '2':
                parts[0] = '0' 
                changed = True
            new_lines.append(" ".join(parts) + "\n")
            
        if changed:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            fixed_count += 1
            
    print(f"[+] Successfully normalized {fixed_count} label files!")
    print("[+] Class 2 merged into Class 0. Ready for retraining.")

if __name__ == "__main__":
    normalize_dataset()