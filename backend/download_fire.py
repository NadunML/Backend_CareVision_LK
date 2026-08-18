# CareVision LK - Model Downloader
# Downloads the pre-trained YOLOv8 model for fire detection

import urllib.request
import urllib.error
import sys

def download_fire_model():
    print("Starting download for YOLOv8 model...")
    
    url = "https://github.com/nimradev064/Real-Time-Fire-Detection-Flask-App/raw/main/Models/best.pt"
    output_filename = "fire_model.pt"

    # Add User-Agent to avoid 403 Forbidden errors
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    try:
        with urllib.request.urlopen(req) as response, open(output_filename, 'wb') as out_file:
            print(f"Downloading to '{output_filename}'...")
            data = response.read()
            out_file.write(data)
            
        print(f"Download complete: {output_filename}")
        
    except urllib.error.URLError as e:
        print(f"Network error while downloading: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_fire_model()