"""
CareVision LK - AI Model Downloader Utility
Fetches the pre-trained YOLOv8 Fire & Smoke detection model.
"""
import urllib.request
import urllib.error
import sys

def download_fire_model():
    print("[INFO] Initiating download: YOLOv8 Fire & Smoke AI Model...")
    
    # Source URL for the pre-trained YOLOv8 weights
    url = "https://github.com/nimradev064/Real-Time-Fire-Detection-Flask-App/raw/main/Models/best.pt"
    output_filename = "fire_model.pt"

    # Utilizing a User-Agent header to bypass potential HTTP 403 restrictions
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    try:
        with urllib.request.urlopen(req) as response, open(output_filename, 'wb') as out_file:
            print(f"[INFO] Connection established. Downloading to '{output_filename}' (This may take a moment)...")
            data = response.read()
            out_file.write(data)
        print(f"✅ [SUCCESS] Model successfully downloaded and saved as '{output_filename}'.")
        
    except urllib.error.URLError as e:
        print(f"❌ [ERROR] Failed to download the model. Network Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ [ERROR] An unexpected operational error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_fire_model()