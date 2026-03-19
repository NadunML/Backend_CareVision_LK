from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

current_mode = 'monitoring'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera_url = os.getenv("CAMERA_URL")

def generate_frames():
    camera = cv2.VideoCapture(camera_url)
    frame_counter = 0
    last_faces = []
    last_mask_status = []
    fire_detected = False

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_counter += 1
            if frame_counter % 5 == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                if current_mode == 'patient':
                    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    last_faces = []
                    for (x, y, w, h) in faces:
                        last_faces.append((x * 2, y * 2, w * 2, h * 2))
                        
                elif current_mode == 'hygiene':
                    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    last_mask_status = []
                    for (x, y, w, h) in faces:
                        orig_x, orig_y, orig_w, orig_h = x * 2, y * 2, w * 2, h * 2
                        roi_y = y + h // 2
                        roi_h = h // 2
                        roi = small_frame[roi_y:roi_y+roi_h, x:x+w]
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        
                        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                        mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)
                        skin_pixels = cv2.countNonZero(mask)
                        total_pixels = w * roi_h
                        skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
                        
                        last_mask_status.append((orig_x, orig_y, orig_w, orig_h, skin_ratio > 0.15))
                        
                elif current_mode == 'fire':
                    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                    lower_fire = np.array([10, 100, 200], dtype=np.uint8)
                    upper_fire = np.array([35, 255, 255], dtype=np.uint8)
                    mask = cv2.inRange(hsv, lower_fire, upper_fire)
                    fire_pixels = cv2.countNonZero(mask)
                    fire_detected = fire_pixels > 1250 # 5000 / 4 due to fx=0.5, fy=0.5
            
            # Draw from cache every frame
            if current_mode == 'patient':
                for (x, y, w, h) in last_faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, 'ALERT: PATIENT WANDERING', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif current_mode == 'hygiene':
                for (x, y, w, h, no_mask) in last_mask_status:
                    if no_mask:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, 'No Mask', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, 'Mask Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif current_mode == 'fire':
                if fire_detected:
                    cv2.putText(frame, 'FIRE DETECTED!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
async def root():
    return {"status": "System Operational"}

@app.post("/set_mode/{mode}")
async def set_mode(mode: str):
    global current_mode
    current_mode = mode
    return {"status": "success", "mode": current_mode}

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
