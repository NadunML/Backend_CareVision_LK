from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
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

modes = {
    'patientIdent': False,
    'maskDetect': False,
    'fireDetect': False
}

# 🔥 කැමරා 9ක් සඳහා ඉඩ වෙන් කිරීම 🔥
camera_urls = {str(i): '' for i in range(1, 10)}

known_face_encodings = []
known_face_names = []
maskNet = None  
fire_model = None 

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
            
            if frame_counter % 4 == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # 🔥 1. Patient Detection (Cameras 1, 2, 3) 🔥
                if cam_id in ['1', '2', '3'] and modes['patientIdent']:
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
                    except Exception:
                        pass

                # 🔥 2. Mask Detection (Cameras 4, 5, 6) 🔥
                elif cam_id in ['4', '5', '6'] and modes['maskDetect']:
                    try:
                        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray_small, 1.1, 4)
                        
                        temp_mask_status = []
                        for (x, y, w, h) in faces:
                            orig_x, orig_y, orig_w, orig_h = x * 4, y * 4, w * 4, h * 4
                            
                            if w < 20 or h < 20:
                                continue

                            if maskNet is not None:
                                face_img = small_frame[y:y+h, x:x+w]
                                if face_img.size > 0:
                                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                    face_img = cv2.resize(face_img, (224, 224))
                                    face_img = img_to_array(face_img)
                                    face_img = preprocess_input(face_img)
                                    face_img = np.expand_dims(face_img, axis=0)
                                    
                                    predictions = maskNet.predict(face_img, verbose=0)[0]
                                    mask_prob = predictions[0]
                                    without_mask_prob = predictions[1]
                                    
                                    no_mask = without_mask_prob > mask_prob
                                else:
                                    no_mask = True
                            else:
                                no_mask = True 
                                
                            temp_mask_status.append((orig_x, orig_y, orig_w, orig_h, no_mask))
                                
                        last_mask_status = temp_mask_status
                    except Exception:
                        pass

                # 🔥 3. Fire Detection (Cameras 7, 8, 9) 🔥
                elif cam_id in ['7', '8', '9'] and modes['fireDetect']:
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
                                    
                                    # Threshold එක 0.35 කරලා තියෙන්නේ Flickering අඩු කරන්න
                                    if 'FIRE' in class_name and conf > 0.35:
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        orig_x1, orig_y1, orig_x2, orig_y2 = x1 * 4, y1 * 4, x2 * 4, y2 * 4
                                        temp_fire_status.append((orig_x1, orig_y1, orig_x2, orig_y2, conf, class_name))
                                        
                        last_fire_status = temp_fire_status
                    except Exception:
                        pass

            # Drawing Bounding Boxes
            if cam_id in ['1', '2', '3'] and modes['patientIdent']:
                for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    if name != "Unknown":
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, f"PT {name}", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, "Staff", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            elif cam_id in ['4', '5', '6'] and modes['maskDetect']:
                for (x, y, w, h, no_mask) in last_mask_status:
                    if no_mask:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3) 
                        cv2.putText(frame, 'ALERT: NO MASK', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
                        cv2.putText(frame, 'Mask Detected', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
            
            elif cam_id in ['7', '8', '9'] and modes['fireDetect']:
                if len(last_fire_status) > 0:
                    cv2.putText(frame, 'EMERGENCY: FIRE DETECTED!', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                    for (x1, y1, x2, y2, conf, class_name) in last_fire_status:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                        cv2.putText(frame, f'{class_name} {int(conf*100)}%', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)

    finally:
        camera.stop() 

# ========================================================
# API Endpoints
# ========================================================
@app.get("/")
async def root():
    return {"status": "Backend & AI Operational"}

@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: str):
    return StreamingResponse(generate_frames(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/get_modes")
async def get_modes():
    return modes

@app.post("/toggle_mode/{mode}")
async def toggle_mode(mode: str):
    global modes
    if mode in modes:
        modes[mode] = not modes[mode]
        return {"status": "success", "modes": modes}
    return {"status": "error"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)