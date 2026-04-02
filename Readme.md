# 🧠 CareVision LK - Edge AI Backend Server

This repository contains the core Python/FastAPI backend that powers the CareVision LK Intelligent Hospital Security System. It acts as the central Edge Node, processing multiple live video streams and executing complex Deep Learning models in real-time without cloud-processing latency.

## ⚙️ Core Architecture & Technologies

* **Framework:** **FastAPI** (Python) for high-performance, asynchronous API routing and live video stream broadcasting.
* **Computer Vision:** **OpenCV** with custom `ThreadedCamera` processing to handle multiple RTSP/HTTP IP camera feeds simultaneously without frame-blocking.
* **Database:** **MySQL** via `mysql-connector-python` for secure, structured logging of security events, access control requests, and registered patient data.

## 🤖 AI Inference Engine

The backend seamlessly orchestrates three distinct AI models to ensure hospital safety:
1. **Fire & Hazard Detection:** Powered by **YOLOv8** (Ultralytics) for instantaneous, dynamic threat recognition in critical zones.
2. **PPE / Mask Compliance:** Powered by a lightweight **MobileNetV2** (TensorFlow/Keras) CNN architecture optimized for rapid frame-by-frame analysis.
3. **Patient Identification:** Powered by **face_recognition** (dlib) utilizing 68 facial landmarks for high-accuracy tracking of high-risk patients.

## 🛠️ Local Setup & Installation

### Prerequisites
* Python 3.9+
* MySQL Server running locally
* A compatible C++ compiler (required for `dlib` / `face_recognition` building)

### 1. Install Dependencies
Clone the repository and install the required Python packages:
```bash
pip install -r requirements.txt