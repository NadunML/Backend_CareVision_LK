from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app) 

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'carevision_db'
}

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/register-patient', methods=['POST'])
def register_patient():
    try:
        patient_id = request.form.get('patientId')
        name = request.form.get('name')
        ward = request.form.get('ward')
        ward_id = request.form.get('wardId')
        risk_level = request.form.get('riskLevel')

        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(f"{patient_id}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        sql = "INSERT INTO patients (patient_id, name, ward, ward_id, risk_level, image_path) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (patient_id, name, ward, ward_id, risk_level, filename)
        cursor.execute(sql, val)
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"message": "Patient registered successfully in MySQL!"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

        # --- අලුතින් දාන කෑල්ල මෙතනින් පටන් ගන්නවා ---
@app.route('/api/patients', methods=['GET'])
def get_patients():
    try:
        conn = mysql.connector.connect(**db_config)
        # dictionary=True දැම්මම ඩේටා ටික ලස්සනට JSON විදිහට එනවා
        cursor = conn.cursor(dictionary=True) 
        
        # අලුත්ම අය උඩින්ම පෙන්නන්න ORDER BY id DESC දානවා
        cursor.execute("SELECT patient_id, name, ward, risk_level, DATE(created_at) as registered_date FROM patients ORDER BY id DESC")
        patients = cursor.fetchall()
        
        # දින (Dates) ටික අකුරු (String) බවට පත් කරනවා React එකට තේරෙන්න
        for p in patients:
            p['registered_date'] = str(p['registered_date'])
            
        cursor.close()
        conn.close()

        return jsonify(patients), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# --- අලුතින් දාන කෑල්ල මෙතනින් ඉවරයි ---

if __name__ == '__main__':
    app.run(debug=True, port=5000)

if __name__ == '__main__':
    app.run(debug=True, port=5000)