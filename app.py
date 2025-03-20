from flask import Flask, render_template, Response, jsonify, request
import cv2
import dlib
import numpy as np
import os
from datetime import datetime
import json
import base64

app = Flask(__name__)

# Configuración
CALIBRATION_FACTOR = float(os.getenv('CALIBRATION_FACTOR', 0.05))
PREDICTOR_PATH = os.getenv('PREDICTOR_PATH', 'shape_predictor_68_face_landmarks.dat')
MEASUREMENTS_FILE = 'mediciones.json'

# Verificación de modelo
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f'Modelo no encontrado: {PREDICTOR_PATH}')

# Inicialización de modelos
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Gestión de cámara (context manager)
class Camera:
    def __enter__(self):
        self.cap = cv2.VideoCapture(0)
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

# Variables seguras
latest_measurements = {}

def calcular_distancia(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calcular_centro(puntos):
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    return (int(np.mean(xs)), int(np.mean(ys)))

def generate_frames():
    global latest_measurements
    with Camera() as cam:
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            # Obtener dimensiones del frame
            height, width = frame.shape[:2]
            
            # Dibujar rectángulo guía
            rect_width = int(width * 0.6)
            rect_height = int(rect_width * 1.5)
            rect_x = (width - rect_width) // 2
            rect_y = int(height * 0.15)
            
            cv2.rectangle(frame, 
                        (rect_x, rect_y),
                        (rect_x + rect_width, rect_y + rect_height),
                        (144, 238, 144),  # Verde claro
                        2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            for face in faces:
                landmarks = predictor(gray, face)
                puntos = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
                
                # Cálculos
                ojo_izq = calcular_centro(puntos[36:42])
                ojo_der = calcular_centro(puntos[42:48])
                nariz = puntos[30]
                
                # Distancias en cm
                medidas = {
                    'izquierda_nariz': round(calcular_distancia(ojo_izq, nariz) * CALIBRATION_FACTOR, 2),
                    'derecha_nariz': round(calcular_distancia(ojo_der, nariz) * CALIBRATION_FACTOR, 2),
                    'entre_ojos': round(calcular_distancia(ojo_izq, ojo_der) * CALIBRATION_FACTOR, 2),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Actualizar mediciones
                latest_measurements = medidas.copy()
                
                # Dibujar elementos
                cv2.circle(frame, ojo_izq, 3, (255,0,0), -1)
                cv2.circle(frame, ojo_der, 3, (255,0,0), -1)
                cv2.circle(frame, nariz, 3, (0,0,255), -1)
                
                # Mostrar texto
                y_pos = face.top() - 60
                for key, value in medidas.items():
                    if key != 'timestamp':
                        cv2.putText(frame, f"{key.replace('_', ' ')}: {value} cm", 
                                    (face.left(), y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        y_pos += 15

            # Codificar frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remote')
def remote():
    return render_template('remote.html')

@app.route('/api/measure', methods=['POST'])
def measure():
    try:
        # Get base64 image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'error': 'No image data provided'}), 400

        # Decode base64 image
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'status': 'error', 'error': 'Invalid image data'}), 400

        # Process image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if not faces:
            return jsonify({'status': 'error', 'error': 'No face detected'}), 404

        # Get measurements for the first detected face
        face = faces[0]
        landmarks = predictor(gray, face)
        puntos = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        # Calculate measurements
        ojo_izq = calcular_centro(puntos[36:42])
        ojo_der = calcular_centro(puntos[42:48])
        nariz = puntos[30]

        measurements = {
            'left_eye_to_nose': round(calcular_distancia(ojo_izq, nariz) * CALIBRATION_FACTOR, 2),
            'right_eye_to_nose': round(calcular_distancia(ojo_der, nariz) * CALIBRATION_FACTOR, 2),
            'between_eyes': round(calcular_distancia(ojo_izq, ojo_der) * CALIBRATION_FACTOR, 2),
            'timestamp': datetime.now().isoformat()
        }

        return jsonify({
            'status': 'success',
            'measurements': measurements
        })

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/guardar_mediciones', methods=['POST'])
def guardar_mediciones():
    if not latest_measurements:
        return jsonify({'estado': 'error', 'mensaje': 'No hay mediciones disponibles'})
    
    try:
        # Cargar datos existentes
        if os.path.exists(MEASUREMENTS_FILE):
            with open(MEASUREMENTS_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # Agregar nueva medición
        data.append(latest_measurements)
        
        # Guardar
        with open(MEASUREMENTS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({
            'estado': 'éxito',
            'datos': latest_measurements,
            'mensaje': 'Mediciones almacenadas'
        })
    
    except Exception as e:
        return jsonify({'estado': 'error', 'mensaje': str(e)}), 500

@app.route('/obtener_mediciones')
def obtener_mediciones():
    try:
        if os.path.exists(MEASUREMENTS_FILE):
            with open(MEASUREMENTS_FILE, 'r') as f:
                return jsonify(json.load(f))
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Server running at http://172.25.48.1:5000")
    print("Access the application from other devices using the above URL")
    app.run(debug=os.getenv('DEBUG', 'False') == 'True', host='0.0.0.0', port=5000)