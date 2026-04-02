"""
CareVision LK - AI Model Downloader Utility
Fetches the pre-trained MobileNetV2 Face Mask detection model.
"""
import urllib.request
import urllib.error
import sys

def download_mask_model():
    print("[INFO] Initiating download: MobileNetV2 Face Mask AI Model...")
    
    # Source URL for the pre-trained Keras model
    url = "https://github.com/chandrikadeb7/Face-Mask-Detection/raw/master/mask_detector.model"
    output_filename = "mask_detector.h5"

    try:
        print(f"[INFO] Connection established. Downloading to '{output_filename}' (This may take a moment)...")
        # Direct retrieval of the model file
        urllib.request.urlretrieve(url, output_filename) 
        print(f"✅ [SUCCESS] Model successfully downloaded and saved as '{output_filename}'.")
        
    except urllib.error.URLError as e:
        print(f"❌ [ERROR] Failed to download the model. Network Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ [ERROR] An unexpected operational error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_mask_model()