from flask import Flask, render_template, Response, redirect, url_for, flash, request
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
from multicamera_holistic import MultiCameraHolisticBiomechanics
from mech_analysis import BiomechanicalAnalysis

# Create a unified application
app = Flask(__name__,
    static_url_path='',
    static_folder='static',
    template_folder='templates'
)
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui'  # Cambia esto por una clave secreta segura
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuración de Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Modelo simple de usuario
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Usuario y contraseña (en una aplicación real, esto debería estar en una base de datos)
VALID_USERNAME = "abc"
VALID_PASSWORD = "000"  # Cambia esto por tu contraseña deseada

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Recreate the CameraManager class from the original app.py
class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.active_cameras = set()
        self.biomech = MultiCameraHolisticBiomechanics()
        self.biomech_analysis = BiomechanicalAnalysis()
        self.all_landmarks = {}  # Almacena landmarks de todas las cámaras

    def add_camera(self, camera_id):
        if camera_id not in self.cameras:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cameras[camera_id] = {
                    'capture': cap,
                    'holistic': self.biomech.mp_holistic.Holistic(
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                }
                self.active_cameras.add(camera_id)
                return True
        return False

    def get_frame(self, camera_id):
        if camera_id in self.cameras:
            success, frame = self.cameras[camera_id]['capture'].read()
            if success:
                frame = cv2.flip(frame, 1)
                processed_frame, landmarks = self.biomech.process_frame(
                    frame,
                    self.cameras[camera_id]['holistic']
                )

                if landmarks:
                    self.all_landmarks[camera_id] = landmarks

                    if len(self.active_cameras) > 1:
                        merged_landmarks = self.biomech.merge_landmarks(self.all_landmarks)
                        if merged_landmarks:
                            # Actualizar análisis biomecánico y obtener ángulos
                            current_angles = self.biomech_analysis.update_points_from_avatar(merged_landmarks)
                            # Emitir landmarks y ángulos
                            socketio.emit('data_update', {
                                'landmarks': merged_landmarks,
                                'angles': current_angles
                            })
                    else:
                        current_angles = self.biomech_analysis.update_points_from_avatar(landmarks)
                        socketio.emit('data_update', {
                            'landmarks': landmarks,
                            'angles': current_angles
                        })

                return processed_frame
        return None

# Initialize camera manager
camera_manager = CameraManager()


# Rutas de autenticación
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == VALID_USERNAME and password == VALID_PASSWORD:
            user = User(username)
            login_user(user)
            return redirect(url_for('biomechanics_page'))
        else:
            flash('Usuario o contraseña incorrectos')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing_page'))


# Rutas principales
@app.route('/')
def landing_page():
    return render_template('landing_page.html')


@app.route('/biomechanics')
@login_required  # Esta decoración protege la ruta
def biomechanics_page():
    return render_template('index.html')


@app.route('/biomechanics2d')
def biomechanics_2d_page():
    return render_template('index2.html')

# Original Routes from app.py
def generate_frames(camera_id):
    while True:
        frame = camera_manager.get_frame(camera_id)
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/emg')
def emg_page():
    return render_template('emg2.html')

@app.route('/video_feed/<int:camera_id>')
@login_required  # Proteger también el feed de video
def video_feed(camera_id):
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# SocketIO Events
@socketio.on('start_camera')
def handle_start_camera(data):
    camera_id = int(data.get('camera_id'))
    if camera_manager.add_camera(camera_id):
        emit('camera_started', {'camera_id': camera_id})

# Additional SocketIO events if needed
@socketio.on('connect')
def handle_connect():
    if not current_user.is_authenticated:
        return
    print('Client connected')

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)