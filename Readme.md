# 🧠 CareVision LK - Edge AI Backend Server

This repository contains the core Python/FastAPI backend for the CareVision LK Hospital Security System. It acts as the central edge node, processing multiple live video streams and running machine learning models locally in real-time.

## Core Architecture

* **Framework:** FastAPI for high-performance, asynchronous API routing and live video stream broadcasting (MJPEG).
* **Video Processing:** OpenCV with a custom `ThreadedCamera` implementation to handle multiple RTSP/HTTP IP camera feeds concurrently without frame-blocking.
* **Database:** MySQL (via `mysql-connector-python`) for structured logging of security events, access logs, and patient records.

## Key Features & AI Models

The backend runs three primary AI modules to ensure hospital safety:

1. **Fire Detection & Emergency Lockdown (YOLOv8):** 
   Monitors critical zones for fire and smoke. If a fire is detected, the system triggers a global emergency lockdown, automatically pausing non-essential AI modules to allocate all processing power to the emergency response.

2. **Patient Wandering Detection (dlib / face_recognition):** 
   Utilizes facial landmarks to identify registered high-risk patients across active camera feeds. Triggers immediate alerts if a patient wanders into unauthorized or unsafe areas.

3. **Mask Compliance Monitoring (MobileNetV2 / Keras):** 
   A lightweight CNN that analyzes faces frame-by-frame to enforce PPE compliance at ward access points.

4. **Edge Security & Resource Management:** 
   Features a secure teardown process. When an admin logs out from the frontend, the backend automatically terminates all active camera connections and clears stream URLs from memory to prevent unauthorized viewing.

## Local Setup & Installation

### Prerequisites

* Python 3.9 or higher
* MySQL Server running locally
* A compatible C++ compiler (required for building `dlib` / `face_recognition`)

### Setup Instructions & How to Run the Server

1. **Clone the repository and navigate to the backend directory.**

2. **Create and activate a virtual environment:**

