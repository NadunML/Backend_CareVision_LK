# CareVision LK - Model Downloader
# Downloads the pre-trained MobileNetV2 mask detection model

import urllib.request
import urllib.error
import sys

def download_mask_model():
    print("Starting download for MobileNetV2 mask model...")
    
    url = "https://github.com/chandrikadeb7/Face-Mask-Detection/raw/master/mask_detector.model"
    output_filename = "mask_detector.h5"

    try:
        print(f"Downloading to '{output_filename}'...")
        urllib.request.urlretrieve(url, output_filename) 
        print(f"Download complete: {output_filename}")
        
    except urllib.error.URLError as e:
        print(f"Network error while downloading: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_mask_model()