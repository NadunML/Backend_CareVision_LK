"""
CareVision LK - Edge AI Backend
Utility script to fetch and verify the YOLOv8 Fire Detection Model.
Author: Shiwon Sachintha
"""

import urllib.request
import urllib.error
import sys
import logging

# Configure professional logging for the backend
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def fetch_fire_detection_model() -> None:
    """
    Downloads the pre-trained YOLOv8 weights required for the fire detection 
    module. Implements custom headers to bypass restrictive firewalls.
    """
    model_url = "https://github.com/nimradev064/Real-Time-Fire-Detection-Flask-App/raw/main/Models/best.pt"
    output_path = "fire_model.pt"

    logging.info("Initializing download sequence for YOLOv8 Fire Detection Model...")

    # Implement User-Agent to prevent HTTP 403 Forbidden responses
    request_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(model_url, headers=request_headers)

    try:
        logging.info("Establishing connection to remote repository...")
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            logging.info(f"Downloading stream to '{output_path}'. This may take a moment...")
            file_data = response.read()
            out_file.write(file_data)
            
        logging.info(f"Successfully retrieved and saved model to: {output_path}")
        
    except urllib.error.URLError as network_error:
        logging.error(f"Network exception encountered during download: {network_error}")
        sys.exit(1)
    except Exception as critical_error:
        logging.critical(f"Unexpected system error occurred: {critical_error}")
        sys.exit(1)

if __name__ == "__main__":
    fetch_fire_detection_model()