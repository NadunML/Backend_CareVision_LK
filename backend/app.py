# CareVision LK Backend API Server

import os
import threading
import time
from datetime import datetime
import cv2
import face_recognition
import mysql.connector
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
from werkzeug.utils import secure_filename

load_dotenv()

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Try loading keras based on availability
try:
    from tf_keras.applications.mobilenet_v2 import preprocess_input
    from tf_keras.preprocessing.image import img_to_array
    from tf_keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.models import load_model
        TF_AVAILABLE = True
    except ImportError:
        print("WARNING: TensorFlow is not installed. Mask detection will be unavailable.")
        TF_AVAILABLE = False

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = FastAPI()
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'carevision_db')
}

# Check db connection on startup
try:
    conn = mysql.connector.connect(**db_config)
    conn.close()
    print("Successfully connected to the database.")
except Exception as e:
    print(f"Database Connection Error: {e}")

camera_ai_configs = {
    str(i): {'patient': False, 'mask': False, 'fire': False} for i in range(1, 10)
}

fire_emergency_active = False

# Initialized empty so no camera turns on automatically
camera_urls = {'1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': '', '9': ''}

known_face_encodings = []
known_face_names = []
maskNet = None  
fire_model = None 

last_log_times = {}
in_memory_patients = []
patient_names_cache = {}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def load_ai_models():
    # Load patient encodings and ai models
    global known_face_encodings, known_face_names, maskNet, fire_model, patient_names_cache, in_memory_patients
    known_face_encodings = []
    known_face_names = []
    patient_names_cache = {}
    
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT patient_id, name FROM patients")
        for row in cursor.fetchall():
            patient_names_cache[str(row['patient_id'])] = row['name']
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Could not load patient names from DB: {e}")
        for p in in_memory_patients:
            patient_names_cache[str(p.get('patient_id', ''))] = p.get('name', 'Unknown')
    
    print("Loading Patient Images...")
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                try:
                    patient_id = filename.split('_')[0] 
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(patient_id)
                except Exception:
                    pass
    
    if TF_AVAILABLE and os.path.exists("mask_detector.h5"):
        print("Loading Deep Learning Mask Model (MobileNetV2)...")
        try:
            maskNet = load_model("mask_detector.h5")
            print("Mask AI loaded successfully.")
        except Exception:
            pass

    if os.path.exists("fire_model.pt"):
        print("Loading YOLOv8 Fire Detection Model...")
        try:
            fire_model = YOLO("fire_model.pt")
            print("Fire AI loaded successfully.")
        except Exception:
            pass

load_ai_models()

class ThreadedCamera:
    # Handles continuous camera reading in a background thread
    def __init__(self, src):
        self.src = src
        self.capture = None
        self.status = False
        self.frame = None
        self.started = False
        self.thread = None
        self._open_capture()

    def _open_capture(self):
        try:
            if self.capture is not None:
                try:
                    self.capture.release()
                except Exception:
                    pass
                self.capture = None

            if isinstance(self.src, int):
                self.capture = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
                if not self.capture.isOpened():
                    self.capture = cv2.VideoCapture(self.src)
            else:
                self.capture = cv2.VideoCapture(self.src)

            if self.capture.isOpened():
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.status, self.frame = self.capture.read()
            else:
                self.status = False
        except Exception:
            self.status = False

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        last_reconnect = time.time()
        while self.started:
            if self.capture is None or not self.capture.isOpened():
                self.status = False
                if time.time() - last_reconnect > 2.0:
                    last_reconnect = time.time()
                    self._open_capture()
                time.sleep(0.1)
                continue

            success, frame = self.capture.read()
            if success and frame is not None:
                self.status = True
                self.frame = frame
            else:
                self.status = False
                # Reconnect logic if stream drops
                if isinstance(self.src, str) and not (self.src.startswith("http://") or self.src.startswith("https://") or self.src.startswith("rtsp://")):
                    try:
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except Exception:
                        pass
                else:
                    if time.time() - last_reconnect > 2.0:
                        last_reconnect = time.time()
                        self._open_capture()
                time.sleep(0.03)

    def read(self):
        return self.status, self.frame

    def stop(self):
        self.started = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.capture:
            try:
                self.capture.release()
            except Exception:
                pass

class CameraRegistry:
    # Manages shared camera instances
    def __init__(self):
        self.lock = threading.Lock()
        self.cameras = {}

    def get_camera(self, src):
        with self.lock:
            if src in self.cameras:
                self.cameras[src]['ref_count'] += 1
                return self.cameras[src]['camera']
            else:
                camera = ThreadedCamera(src).start()
                self.cameras[src] = {
                    'camera': camera,
                    'ref_count': 1
                }
                return camera

    def release_camera(self, src):
        with self.lock:
            if src in self.cameras:
                self.cameras[src]['ref_count'] -= 1
                if self.cameras[src]['ref_count'] <= 0:
                    camera = self.cameras[src]['camera']
                    camera.stop()
                    del self.cameras[src]

camera_registry = CameraRegistry()

def generate_frames(cam_id):
    # Main video pipeline with AI processing
    global last_log_times, fire_emergency_active, patient_names_cache
    current_src = None
    camera = None
    frame_counter = 0

    last_face_locations = []
    last_face_names = []
    last_mask_status = []
    last_fire_status = []
    
    fire_consecutive_count = 0

    try:
        while True:
            raw_url = camera_urls.get(cam_id, '').strip()

            if not raw_url:
                if camera is not None and current_src is not None:
                    camera_registry.release_camera(current_src)
                    camera = None
                    current_src = None

                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, f"CAM 0{cam_id}: NO SIGNAL", (180, 220), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 100, 100), 2)
                cv2.putText(blank_frame, "Configure URL in Settings", (170, 260), cv2.FONT_HERSHEY_DUPLEX, 0.6, (70, 70, 70), 1)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.2)
                continue

            if raw_url.isdigit():
                cam_source = int(raw_url)
            else:
                cam_source = raw_url

            if current_src != cam_source:
                if camera is not None and current_src is not None:
                    camera_registry.release_camera(current_src)
                current_src = cam_source
                camera = camera_registry.get_camera(cam_source)

            success, frame = camera.read()
            if not success or frame is None:
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, f"CAM 0{cam_id}: CONNECTING...", (160, 240), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 165, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue
                
            frame = frame.copy() 
            frame_counter += 1
            
            cam_config = camera_ai_configs.get(cam_id, {'patient': False, 'mask': False, 'fire': False})
            
            if not cam_config['fire']:
                last_fire_status.clear()
                fire_consecutive_count = 0
                
            if fire_emergency_active:
                last_face_locations.clear()
                last_face_names.clear()
                last_mask_status.clear()
            else:
                if not cam_config['patient']:
                    last_face_locations.clear()
                    last_face_names.clear()
                if not cam_config['mask']:
                    last_mask_status.clear()

            h, w = frame.shape[:2]
            target_w = min(w, 320)
            scale = w / max(target_w, 1)
            
            # Fire detection
            if cam_config['fire'] and frame_counter % 5 == 0:
                small_frame_fire = cv2.resize(frame, (target_w, int(h / max(scale, 1e-6))))
                try:
                    temp_fire_status = []
                    fire_detected_this_frame = False

                    if fire_model is not None:
                        results = fire_model(small_frame_fire, verbose=False)
                        for r in results:
                            for box in r.boxes:
                                conf = float(box.conf[0])
                                cls_id = int(box.cls[0])
                                class_name = fire_model.names[cls_id].upper()

                                if 'FIRE' in class_name and 'SMOKE' not in class_name and conf > 0.45:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    
                                    roi = small_frame_fire[y1:y2, x1:x2]
                                    if roi.size > 0:
                                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                        lower_fire = np.array([0, 40, 150]) 
                                        upper_fire = np.array([35, 255, 255])
                                        mask = cv2.inRange(hsv, lower_fire, upper_fire)
                                        
                                        fire_pixel_ratio = np.sum(mask > 0) / (roi.size / 3 + 1e-6)
                                        
                                        if fire_pixel_ratio > 0.02:
                                            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                                            temp_fire_status.append((orig_x1, orig_y1, orig_x2, orig_y2, conf, class_name))
                                            fire_detected_this_frame = True

                    if fire_detected_this_frame:
                        fire_consecutive_count += 1
                    else:
                        fire_consecutive_count = 0  

                    if fire_detected_this_frame and fire_consecutive_count >= 1:
                        fire_emergency_active = True 

                        current_time = time.time()
                        fire_log_key = f"fire_{cam_id}"
                        if current_time - last_log_times.get(fire_log_key, 0) > 10:
                            try:
                                db_conn = mysql.connector.connect(**db_config)
                                db_cursor = db_conn.cursor()
                                db_cursor.execute("INSERT INTO fire_logs (camera_id, status) VALUES (%s, 'Active')", (f"Cam 0{cam_id}",))
                                db_cursor.execute("INSERT INTO system_alerts (alert_type, camera_id) VALUES (%s, %s)", ('Fire', f"Cam 0{cam_id}"))
                                db_conn.commit()
                                db_cursor.close()
                                db_conn.close()
                                last_log_times[fire_log_key] = current_time
                            except Exception:
                                pass

                    last_fire_status = temp_fire_status if fire_detected_this_frame else []
                except Exception:
                    pass

            # Patient and mask detection
            if frame_counter % 30 == 0:
                small_frame = cv2.resize(frame, (target_w, int(h / max(scale, 1e-6))))

                if cam_config['patient'] and not fire_emergency_active:
                    try:
                        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        final_ai_frame = np.array(rgb_small_frame, dtype=np.uint8).copy()
                        
                        last_face_locations = face_recognition.face_locations(final_ai_frame)
                        face_encodings = face_recognition.face_encodings(final_ai_frame, last_face_locations)

                        last_face_names = []
                        for face_encoding in face_encodings:
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                            name = "Unknown"
                            if True in matches:
                                first_match_index = matches.index(True)
                                name = known_face_names[first_match_index]
                            last_face_names.append(name)
                            
                            if name != "Unknown":
                                current_time = time.time()
                                alert_key = f"pat_{cam_id}_{name}"
                                if current_time - last_log_times.get(alert_key, 0) > 20: 
                                    try:
                                        full_name = patient_names_cache.get(name, name)
                                        alert_text = f"Patient Wandering - {full_name}"
                                        
                                        db_conn = mysql.connector.connect(**db_config)
                                        db_cursor = db_conn.cursor()
                                        db_cursor.execute("INSERT INTO system_alerts (alert_type, camera_id) VALUES (%s, %s)", (alert_text, f"Cam 0{cam_id}"))
                                        db_conn.commit()
                                        db_cursor.close()
                                        db_conn.close()
                                        last_log_times[alert_key] = current_time
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                if cam_config['mask'] and not fire_emergency_active:
                    try:
                        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=4)
                        
                        if len(faces) == 0:
                            eyes = eye_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5)
                            if len(eyes) > 0:
                                (ex, ey, ew, eh) = eyes[0]
                                fx = max(0, ex - int(ew * 1.5))
                                fy = max(0, ey - int(eh * 1.5))
                                fw = int(ew * 4.5)
                                fh = int(eh * 5.5)
                                faces = [[fx, fy, fw, fh]]

                        if len(faces) > 0:
                            temp_mask_status = []
                            for (x, y, w, h) in faces:
                                orig_x, orig_y, orig_w, orig_h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
                                if w < 20 or h < 20: continue

                                no_mask = False
                                if maskNet is not None:
                                    face_img = small_frame[y:y+h, x:x+w]
                                    if face_img.size > 0:
                                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                        face_img = cv2.resize(face_img, (224, 224))
                                        face_img = img_to_array(face_img)
                                        face_img = preprocess_input(face_img)
                                        face_img = np.expand_dims(face_img, axis=0)
                                        predictions = maskNet.predict(face_img, verbose=0)[0]
                                        mask_prob, without_mask_prob = predictions[0], predictions[1]
                                        no_mask = without_mask_prob > mask_prob
                                        
                                        current_time = time.time()
                                        if current_time - last_log_times.get(cam_id, 0) > 10: 
                                            mask_det_str = "No" if no_mask else "Yes"
                                            acc_res_str = "Denied" if no_mask else "Granted"
                                            try:
                                                db_conn = mysql.connector.connect(**db_config)
                                                db_cursor = db_conn.cursor()
                                                db_cursor.execute("INSERT INTO access_logs (camera_id, mask_detected, access_result) VALUES (%s, %s, %s)", (f"Cam 0{cam_id}", mask_det_str, acc_res_str))
                                                if no_mask:
                                                    db_cursor.execute("INSERT INTO system_alerts (alert_type, camera_id) VALUES (%s, %s)", ('Mask Violation', f"Cam 0{cam_id}"))
                                                db_conn.commit()
                                                db_cursor.close()
                                                db_conn.close()
                                                last_log_times[cam_id] = current_time
                                            except Exception:
                                                pass
                                else: 
                                    no_mask = True 
                                
                                temp_mask_status.append((orig_x, orig_y, orig_w, orig_h, no_mask))
                            
                            last_mask_status = temp_mask_status
                        else:
                            last_mask_status = []
                                
                    except Exception:
                        pass

            #  UI overlays
            if cam_config['patient'] and not fire_emergency_active:
                for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
                    t, r, b, l = int(top * scale), int(right * scale), int(bottom * scale), int(left * scale)
                    if name != "Unknown":
                        cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
                        cv2.putText(frame, f"PT {name}", (l + 6, b - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

            if cam_config['mask'] and not fire_emergency_active:
                for (x, y, w, h, no_mask) in last_mask_status:
                    if no_mask:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3) 
                        cv2.putText(frame, 'ALERT: NO MASK', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
                        cv2.putText(frame, 'Mask Detected', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
            
            if cam_config['fire'] or fire_emergency_active:
                if len(last_fire_status) > 0:
                    cv2.putText(frame, 'EMERGENCY: FIRE DETECTED!', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                    for (x1, y1, x2, y2, conf, class_name) in last_fire_status:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                        cv2.putText(frame, f'{class_name} {int(conf*100)}%', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                elif fire_emergency_active:
                    cv2.putText(frame, 'SYSTEM LOCKDOWN: FIRE DETECTED', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 165, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.04)

    finally:
        if camera is not None and current_src is not None:
            camera_registry.release_camera(current_src) 

class AIConfigUpdate(BaseModel):
    camId: str
    module: str
    status: bool

class LoginHistoryRequest(BaseModel):
    email: str

@app.get("/")
async def root():
    return {"status": "Backend & AI Operational"}

@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: str):
    return StreamingResponse(generate_frames(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/set_camera_ai")
async def set_camera_ai(config: AIConfigUpdate):
    global camera_ai_configs
    if config.camId in camera_ai_configs and config.module in camera_ai_configs[config.camId]:
        camera_ai_configs[config.camId][config.module] = config.status
        return {"status": "success", "configs": camera_ai_configs}
    return {"status": "error"}

@app.post("/api/disable_non_fire")
async def disable_non_fire():
    global camera_ai_configs
    for cam in camera_ai_configs:
        camera_ai_configs[cam].update({'patient': False, 'mask': False})
    return {"status": "success", "configs": camera_ai_configs}

@app.get("/api/get_camera_ai")
async def get_camera_ai():
    return camera_ai_configs

@app.get("/api/emergency-status")
async def get_emergency_status():
    global fire_emergency_active
    return {"emergency_lockdown": fire_emergency_active}

@app.get("/api/get_modes")
async def get_modes():
    return camera_ai_configs.get('1', {})

@app.post("/toggle_mode/{feature}")
async def toggle_mode(feature: str):
    return {"status": "success"}

@app.post("/api/update_camera")
async def update_camera(cam_id: str = Form(...), url: str = Form(default="")):
    camera_urls[cam_id] = url
    return {"status": "success", "camera": cam_id, "url": url}

@app.get("/api/get_cameras")
async def get_cameras():
    return camera_urls

@app.post("/api/disconnect_all_cameras")
async def disconnect_all_cameras():
    # Disconnects all cameras when requested (e.g., on user logout)
    global camera_urls
    for cam_id in camera_urls:
        camera_urls[cam_id] = ""
    return {"status": "success", "message": "All cameras disconnected"}

@app.post("/api/register-patient")
async def register_patient(
    patientId: str = Form(...),
    name: str = Form(...),
    ward: str = Form(...),
    wardId: str = Form(...),
    riskLevel: str = Form(default=""),
    image: UploadFile = File(...)
):
    global in_memory_patients, known_face_encodings, known_face_names, patient_names_cache
    try:
        filename = secure_filename(f"{patientId}_{image.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as buffer:
            buffer.write(await image.read())

        patient_record = {
            "patient_id": patientId,
            "name": name,
            "ward": ward,
            "ward_id": wardId,
            "image_path": filename,
            "registered_date": str(datetime.now().date())
        }

        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            sql = "INSERT INTO patients (patient_id, name, ward, ward_id, image_path) VALUES (%s, %s, %s, %s, %s)"
            val = (patientId, name, ward, wardId, filename)
            cursor.execute(sql, val)
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as db_err:
            in_memory_patients = [p for p in in_memory_patients if p.get('patient_id') != patientId]
            in_memory_patients.insert(0, patient_record)

        # Update AI Memory instantly without restarting full AI models
        patient_names_cache[str(patientId)] = name
        try:
            new_image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(new_image)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(patientId)
        except Exception:
            pass

        return {"status": "success", "message": "Patient registered successfully!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/patients")
async def get_patients():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True) 
        cursor.execute("SELECT patient_id, name, ward, image_path, DATE(created_at) as registered_date FROM patients ORDER BY id DESC")
        patients = cursor.fetchall()
        for p in patients:
            p['registered_date'] = str(p['registered_date'])
        cursor.close()
        conn.close()
        return patients
    except Exception as e:
        print(f"DB Patient Fetch Notice (Using Memory Fallback): {e}")
        return in_memory_patients

@app.delete("/api/delete-patient/{patient_id}")
async def delete_patient(patient_id: str):
    global in_memory_patients, known_face_encodings, known_face_names
    try:
        # 1. Delete from Database
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM patients WHERE patient_id = %s", (patient_id,))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception:
            pass

        # 2. Clear from fallback memory
        in_memory_patients = [p for p in in_memory_patients if p.get('patient_id') != patient_id]

        # 3. Force delete ALL image files belonging to this patient
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if filename.startswith(f"{patient_id}_"):
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass

        # 4. INSTANTLY remove the face from AI active memory tracking
        indices_to_keep = [i for i, name in enumerate(known_face_names) if name != patient_id]
        known_face_encodings = [known_face_encodings[i] for i in indices_to_keep]
        known_face_names = [known_face_names[i] for i in indices_to_keep]

        return {"status": "success", "message": "Patient deleted successfully!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/access_logs")
async def get_access_logs():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True) 
        cursor.execute("SELECT id, camera_id, mask_detected, access_result, timestamp FROM access_logs ORDER BY timestamp DESC LIMIT 50")
        logs = cursor.fetchall()
        for log in logs:
            log['timestamp'] = str(log['timestamp'])
        cursor.close()
        conn.close()
        return logs
    except Exception:
        return []

@app.get("/api/fire_logs")
async def get_fire_logs():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, camera_id, status, timestamp FROM fire_logs ORDER BY timestamp DESC LIMIT 50")
        logs = cursor.fetchall()
        for log in logs:
            log['timestamp'] = str(log['timestamp'])
        cursor.close()
        conn.close()
        return logs
    except Exception:
        return []

@app.post("/api/resolve_fire_alert/{log_id}")
async def resolve_fire_alert(log_id: int):
    global fire_emergency_active
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT camera_id FROM fire_logs WHERE id = %s", (log_id,))
        log = cursor.fetchone()
        
        if log:
            cursor.execute("UPDATE fire_logs SET status = 'Resolved' WHERE id = %s", (log_id,))
            cursor.execute("UPDATE system_alerts SET status = 'Resolved' WHERE camera_id = %s AND alert_type = 'Fire' AND status = 'Pending'", (log['camera_id'],))
            
        conn.commit()
        cursor.execute("SELECT COUNT(*) as cnt FROM fire_logs WHERE status = 'Active'")
        remaining = cursor.fetchone()
        if remaining and remaining['cnt'] == 0:
            fire_emergency_active = False

        cursor.close()
        conn.close()
        return {"status": "success"}
    except Exception:
        return {"status": "error"}

@app.get("/api/system_alerts")
async def get_system_alerts():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True) 
        cursor.execute("SELECT id, alert_type, camera_id, status, timestamp FROM system_alerts ORDER BY timestamp DESC LIMIT 50")
        logs = cursor.fetchall()
        for log in logs:
            log['timestamp'] = str(log['timestamp'])
        cursor.close()
        conn.close()
        return logs
    except Exception:
        return []

@app.post("/api/resolve_system_alert/{alert_id}")
async def resolve_system_alert(alert_id: int):
    global fire_emergency_active
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT alert_type, camera_id FROM system_alerts WHERE id = %s", (alert_id,))
        alert = cursor.fetchone()
        
        if alert:
            cursor.execute("UPDATE system_alerts SET status = 'Resolved' WHERE id = %s", (alert_id,))
            if alert['alert_type'] == 'Fire':
                cursor.execute("UPDATE fire_logs SET status = 'Resolved' WHERE camera_id = %s AND status = 'Active'", (alert['camera_id'],))
        
        conn.commit()
        cursor.execute("SELECT COUNT(*) as cnt FROM fire_logs WHERE status = 'Active'")
        remaining = cursor.fetchone()
        if remaining and remaining['cnt'] == 0:
            fire_emergency_active = False

        cursor.close()
        conn.close()
        return {"status": "success"}
    except Exception:
        return {"status": "error"}

@app.post("/api/login-history")
async def save_login_history(req: LoginHistoryRequest):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO login_history (email) VALUES (%s)", (req.email,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success", "message": "Login history saved!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/login-history")
async def get_login_history():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, email, login_time FROM login_history ORDER BY login_time DESC LIMIT 50")
        records = cursor.fetchall()
        for r in records:
            r['login_time'] = str(r['login_time'])
        cursor.close()
        conn.close()
        return records
    except Exception:
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)