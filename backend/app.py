from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mysql.connector
import cv2
import numpy as np
import os
import face_recognition
from werkzeug.utils import secure_filename
import threading
import time
from ultralytics import YOLO 

os.environ["TF_USE_LEGACY_KERAS"] = "1"

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
        print("⚠️ WARNING: TensorFlow is not installed properly!")
        TF_AVAILABLE = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'carevision_db'
}

try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            camera_id VARCHAR(50),
            mask_detected VARCHAR(10),
            confidence VARCHAR(10),
            access_result VARCHAR(20),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fire_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            camera_id VARCHAR(50),
            event_type VARCHAR(50),
            confidence VARCHAR(10),
            severity VARCHAR(20),
            status VARCHAR(20) DEFAULT 'Active',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_alerts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            alert_type VARCHAR(50),
            description VARCHAR(255),
            camera_id VARCHAR(50),
            priority VARCHAR(20),
            status VARCHAR(20) DEFAULT 'Pending',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Updated default dummy data to match Cam 01 - 05
    cursor.execute("SELECT COUNT(*) FROM access_logs")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO access_logs (camera_id, mask_detected, confidence, access_result) VALUES ('Cam 05', 'Yes', '98%', 'Granted'), ('Cam 04', 'No', '95%', 'Denied'), ('Cam 02', 'Yes', '92%', 'Granted')")
    
    cursor.execute("SELECT COUNT(*) FROM fire_logs")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO fire_logs (camera_id, event_type, confidence, severity, status) VALUES ('Cam 01', 'Smoke Detected', '95%', 'High', 'Active'), ('Cam 03', 'Fire Detected', '88%', 'Critical', 'Resolved')")
    
    cursor.execute("SELECT COUNT(*) FROM system_alerts")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO system_alerts (alert_type, description, camera_id, priority, status) VALUES ('Fire', 'Fire detected in Store Room', 'Cam 01', 'High', 'Pending'), ('Patient Wandering', 'Patient John Anderson detected at Exit', 'Cam 02', 'High', 'Pending'), ('Mask Violation', 'Staff entered ICU without proper mask', 'Cam 04', 'Medium', 'Pending')")

    conn.commit()
    cursor.close()
    conn.close()
except Exception as e:
    print(f"DB Init Error: {e}")

# NEW: Nested config for 5 cameras, replacing the old global 'modes'
camera_ai_configs = {
    str(i): {'patient': False, 'mask': False, 'fire': False} for i in range(1, 6)
}

# Adjusted to 5 cameras
camera_urls = {str(i): '' for i in range(1, 6)}

known_face_encodings = []
known_face_names = []
maskNet = None  
fire_model = None 

last_log_times = {}

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_ai_models():
    global known_face_encodings, known_face_names, maskNet, fire_model
    known_face_encodings = []
    known_face_names = []
    
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
                except Exception as e:
                    pass
    
    if TF_AVAILABLE and os.path.exists("mask_detector.h5"):
        print("Loading Deep Learning Mask Model (MobileNetV2)...")
        try:
            maskNet = load_model("mask_detector.h5")
            print("✅ Modern Mask AI Loaded Successfully!")
        except Exception as e:
            pass

    if os.path.exists("fire_model.pt"):
        print("Loading YOLOv8 Fire Detection Model...")
        try:
            fire_model = YOLO("fire_model.pt")
            print("✅ Modern Fire AI (YOLOv8) Loaded Successfully!")
        except Exception as e:
            pass

load_ai_models()

class ThreadedCamera:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.status, self.frame = self.capture.read()
        self.started = False
        self.thread = None

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            success, frame = self.capture.read()
            if success:
                self.status, self.frame = success, frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.status, self.frame

    def stop(self):
        self.started = False
        if self.thread:
            self.thread.join()
        self.capture.release()

def generate_frames(cam_id):
    global last_log_times
    url = camera_urls.get(cam_id, '')
    if not url:
        return

    if url.isdigit():
        cam_source = int(url)
    else:
        cam_source = url

    camera = ThreadedCamera(cam_source).start()
    frame_counter = 0

    last_face_locations = []
    last_face_names = []
    last_mask_status = []
    last_fire_status = [] 

    try:
        while True:
            if camera_urls.get(cam_id) != url:
                break

            success, frame = camera.read()
            if not success or frame is None:
                time.sleep(0.05)
                continue
                
            frame = frame.copy() 
            frame_counter += 1
            
            # Fetch the specific AI configuration for this camera
            cam_config = camera_ai_configs.get(cam_id, {'patient': False, 'mask': False, 'fire': False})
            
            if frame_counter % 15 == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Patient Detection Logic
                if cam_config['patient']:
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
                                        db_conn = mysql.connector.connect(**db_config)
                                        db_cursor = db_conn.cursor()
                                        db_cursor.execute(
                                            "INSERT INTO system_alerts (alert_type, description, camera_id, priority) VALUES (%s, %s, %s, %s)",
                                            ('Patient Wandering', f'Patient {name} detected at unauthorized zone', f"Cam 0{cam_id}", 'High')
                                        )
                                        db_conn.commit()
                                        db_cursor.close()
                                        db_conn.close()
                                        last_log_times[alert_key] = current_time
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                # Mask Detection Logic
                if cam_config['mask']:
                    try:
                        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray_small, 1.1, 4)
                        
                        temp_mask_status = []
                        for (x, y, w, h) in faces:
                            orig_x, orig_y, orig_w, orig_h = x * 4, y * 4, w * 4, h * 4
                            if w < 20 or h < 20: continue

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
                                    conf_val = f"{int(max(mask_prob, without_mask_prob) * 100)}%"
                                    
                                    current_time = time.time()
                                    if current_time - last_log_times.get(cam_id, 0) > 10: 
                                        mask_det_str = "No" if no_mask else "Yes"
                                        acc_res_str = "Denied" if no_mask else "Granted"
                                        try:
                                            db_conn = mysql.connector.connect(**db_config)
                                            db_cursor = db_conn.cursor()
                                            db_cursor.execute("INSERT INTO access_logs (camera_id, mask_detected, confidence, access_result) VALUES (%s, %s, %s, %s)", (f"Cam 0{cam_id}", mask_det_str, conf_val, acc_res_str))
                                            if no_mask:
                                                db_cursor.execute("INSERT INTO system_alerts (alert_type, description, camera_id, priority) VALUES (%s, %s, %s, %s)", ('Mask Violation', 'Staff entered zone without proper mask', f"Cam 0{cam_id}", 'Medium'))
                                            db_conn.commit()
                                            db_cursor.close()
                                            db_conn.close()
                                            last_log_times[cam_id] = current_time
                                        except Exception:
                                            pass
                            else: no_mask = True 
                            temp_mask_status.append((orig_x, orig_y, orig_w, orig_h, no_mask))
                        last_mask_status = temp_mask_status
                    except Exception:
                        pass

                # Fire Detection Logic
                if cam_config['fire']:
                    try:
                        temp_fire_status = []
                        if fire_model is not None:
                            results = fire_model(small_frame, verbose=False)
                            for r in results:
                                boxes = r.boxes
                                for box in boxes:
                                    conf = float(box.conf[0])
                                    cls_id = int(box.cls[0])
                                    class_name = fire_model.names[cls_id].upper() 
                                    
                                    if 'FIRE' in class_name and conf > 0.35:
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        orig_x1, orig_y1, orig_x2, orig_y2 = x1 * 4, y1 * 4, x2 * 4, y2 * 4
                                        temp_fire_status.append((orig_x1, orig_y1, orig_x2, orig_y2, conf, class_name))
                                        
                                        current_time = time.time()
                                        fire_log_key = f"fire_{cam_id}"
                                        if current_time - last_log_times.get(fire_log_key, 0) > 15: 
                                            event_type = "Fire Detected" if 'FIRE' in class_name else "Smoke Detected"
                                            severity = "Critical" if conf > 0.65 else "High"
                                            conf_val = f"{int(conf * 100)}%"
                                            try:
                                                db_conn = mysql.connector.connect(**db_config)
                                                db_cursor = db_conn.cursor()
                                                db_cursor.execute("INSERT INTO fire_logs (camera_id, event_type, confidence, severity, status) VALUES (%s, %s, %s, %s, 'Active')", (f"Cam 0{cam_id}", event_type, conf_val, severity))
                                                db_cursor.execute("INSERT INTO system_alerts (alert_type, description, camera_id, priority) VALUES (%s, %s, %s, %s)", ('Fire', f'{event_type} in Danger Zone', f"Cam 0{cam_id}", 'High'))
                                                db_conn.commit()
                                                db_cursor.close()
                                                db_conn.close()
                                                last_log_times[fire_log_key] = current_time
                                            except Exception:
                                                pass
                        last_fire_status = temp_fire_status
                    except Exception:
                        pass

            # Drawing Overlays on the Frame
            if cam_config['patient']:
                for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    if name != "Unknown":
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, f"PT {name}", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, "Staff", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            if cam_config['mask']:
                for (x, y, w, h, no_mask) in last_mask_status:
                    if no_mask:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3) 
                        cv2.putText(frame, 'ALERT: NO MASK', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
                        cv2.putText(frame, 'Mask Detected', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
            
            if cam_config['fire']:
                if len(last_fire_status) > 0:
                    cv2.putText(frame, 'EMERGENCY: FIRE DETECTED!', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                    for (x1, y1, x2, y2, conf, class_name) in last_fire_status:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                        cv2.putText(frame, f'{class_name} {int(conf*100)}%', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.01)

    finally:
        camera.stop() 


# --- API Models and Endpoints ---

class AIConfigUpdate(BaseModel):
    camId: str
    module: str
    status: bool

@app.get("/")
async def root():
    return {"status": "Backend & AI Operational"}

@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: str):
    return StreamingResponse(generate_frames(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

# NEW: Update AI settings for a specific camera
@app.post("/api/set_camera_ai")
async def set_camera_ai(config: AIConfigUpdate):
    global camera_ai_configs
    if config.camId in camera_ai_configs and config.module in camera_ai_configs[config.camId]:
        camera_ai_configs[config.camId][config.module] = config.status
        return {"status": "success", "configs": camera_ai_configs}
    return {"status": "error"}

# Kept for legacy dashboard compatibility
@app.get("/api/get_modes")
async def get_modes():
    return camera_ai_configs.get('1', {})

@app.post("/api/update_camera")
async def update_camera(cam_id: str = Form(...), url: str = Form(default="")):
    camera_urls[cam_id] = url
    return {"status": "success", "camera": cam_id, "url": url}

@app.get("/api/get_cameras")
async def get_cameras():
    return camera_urls

@app.post("/api/register-patient")
async def register_patient(
    patientId: str = Form(...),
    name: str = Form(...),
    ward: str = Form(...),
    wardId: str = Form(...),
    riskLevel: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        filename = secure_filename(f"{patientId}_{image.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as buffer:
            buffer.write(await image.read())

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        sql = "INSERT INTO patients (patient_id, name, ward, ward_id, risk_level, image_path) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (patientId, name, ward, wardId, riskLevel, filename)
        cursor.execute(sql, val)
        conn.commit()
        cursor.close()
        conn.close()

        load_ai_models() 
        return {"message": "Patient registered successfully!"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/patients")
async def get_patients():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True) 
        cursor.execute("SELECT patient_id, name, ward, risk_level, DATE(created_at) as registered_date FROM patients ORDER BY id DESC")
        patients = cursor.fetchall()
        for p in patients:
            p['registered_date'] = str(p['registered_date'])
        cursor.close()
        conn.close()
        return patients
    except Exception as e:
        return {"error": str(e)}

@app.delete("/api/delete-patient/{patient_id}")
async def delete_patient(patient_id: str):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM patients WHERE patient_id = %s", (patient_id,))
        conn.commit()
        cursor.close()
        conn.close()
        load_ai_models() 
        return {"status": "success", "message": "Patient deleted successfully!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/access_logs")
async def get_access_logs():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True) 
        cursor.execute("SELECT camera_id, mask_detected, confidence, access_result, timestamp FROM access_logs ORDER BY timestamp DESC LIMIT 50")
        logs = cursor.fetchall()
        for log in logs:
            log['timestamp'] = str(log['timestamp'])
        cursor.close()
        conn.close()
        return logs
    except Exception as e:
        return []

@app.get("/api/fire_logs")
async def get_fire_logs():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True) 
        cursor.execute("SELECT id, camera_id, event_type, confidence, severity, status, timestamp FROM fire_logs ORDER BY timestamp DESC LIMIT 50")
        logs = cursor.fetchall()
        for log in logs:
            log['timestamp'] = str(log['timestamp'])
        cursor.close()
        conn.close()
        return logs
    except Exception as e:
        return []

@app.post("/api/resolve_fire_alert/{log_id}")
async def resolve_fire_alert(log_id: int):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("UPDATE fire_logs SET status = 'Resolved' WHERE id = %s", (log_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error"}

@app.get("/api/system_alerts")
async def get_system_alerts():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True) 
        cursor.execute("SELECT id, alert_type, description, camera_id, priority, status, timestamp FROM system_alerts ORDER BY timestamp DESC LIMIT 50")
        logs = cursor.fetchall()
        for log in logs:
            log['timestamp'] = str(log['timestamp'])
        cursor.close()
        conn.close()
        return logs
    except Exception as e:
        return []

@app.post("/api/resolve_system_alert/{alert_id}")
async def resolve_system_alert(alert_id: int):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("UPDATE system_alerts SET status = 'Resolved' WHERE id = %s", (alert_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)