from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
import cv2
import numpy as np
import os
import face_recognition
from werkzeug.utils import secure_filename

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

camera_url = 0

known_face_encodings = []
known_face_names = []
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_registered_patients():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    print("Loading patient images for AI...")
    
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
                        print(f"✅ AI Loaded patient ID: {patient_id}")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")

load_registered_patients()


def generate_frames():
    camera = cv2.VideoCapture(camera_url)
    frame_counter = 0

    last_face_locations = []
    last_face_names = []

    while True:
        success, frame = camera.read()
        if not success or frame is None:
            break
        else:
            frame_counter += 1
            
            if modes['patientIdent']:
                if frame_counter % 4 == 0:
                    try:
                        # 1. රූපය පොඩි කරනවා
                        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                        
                        # 2. BGR (OpenCV) වලින් RGB වලට හරවනවා
                        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        
                        # ==========================================================
                        # 🔥 THE BULLETPROOF FIX 🔥
                        # මේකෙන් රූපය අනිවාර්යයෙන්ම 8-bit unsigned integer (uint8) කරනවා
                        # සහ Memory එකේ අලුත් copy එකක් හදනවා (Contiguous)
                        # ==========================================================
                        final_ai_frame = np.array(rgb_small_frame, dtype=np.uint8).copy()
                        
                        # 3. මූණ හොයනවා
                        last_face_locations = face_recognition.face_locations(final_ai_frame)
                        
                        # 4. මූණේ ලක්ෂණ ගන්නවා
                        face_encodings = face_recognition.face_encodings(final_ai_frame, last_face_locations)

                        last_face_names = []
                        for face_encoding in face_encodings:
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                            name = "Unknown"

                            if True in matches:
                                first_match_index = matches.index(True)
                                name = known_face_names[first_match_index]

                            last_face_names.append(name)
                            
                    except Exception as e:
                        # එරර් එකක් ආවොත් Terminal එකේ රතු පාටින් ප්‍රින්ට් කරන්නෙ නැතුව මෙහෙම දානවා
                        print(f"⚠️ Video Frame Error (Skipped): {e}")
                        pass

                for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
                    top *= 4; right *= 4; bottom *= 4; left *= 4

                    if name != "Unknown":
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        cv2.putText(frame, f"ALERT: PT {name}", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    else:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, "Staff/Unknown", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/")
async def root():
    return {"status": "Backend & AI Operational"}

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/toggle_mode/{mode}")
async def toggle_mode(mode: str):
    global modes
    if mode in modes:
        modes[mode] = not modes[mode]
        return {"status": "success", "modes": modes}
    return {"status": "error", "message": "Invalid mode"}

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

        load_registered_patients()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)