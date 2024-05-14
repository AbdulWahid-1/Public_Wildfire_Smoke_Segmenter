import urllib.request
import os

def download_test_suite():
    # Direct download endpoints for the 4 hard-negative Unsplash images
    urls = {
        "test_red_roof.jpg": "https://unsplash.com/photos/hcG8Fg8Eurk/download?force=true",
        "test_sunset.jpg": "https://unsplash.com/photos/247tL7bLJdY/download?force=true",
        "test_autumn.jpg": "https://unsplash.com/photos/moHqJmdANys/download?force=true",
        "test_fireplace.jpg": "https://unsplash.com/photos/JfvtxuPBPCc/download?force=true"
    }

    print("[*] Downloading Hard-Negative Test Suite...")
    
    # We use a standard browser User-Agent so Unsplash doesn't block the request
    req_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for filename, url in urls.items():
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req) as response:
                with open(filename, 'wb') as f:
                    f.write(response.read())
            print(f"  [+] Saved: {filename}")
        except Exception as e:
            print(f"  [-] Failed to download {filename}: {e}")

    print("\n[+] Download complete. You can now test them!")
    print("[*] Example: python inference_det.py test_red_roof.jpg")

if __name__ == "__main__":
    download_test_suite()