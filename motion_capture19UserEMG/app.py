# motion_capture
# ├── app.py                      # Backend principal (FastAPI + WebSocket)
# ├── multicamera_holistic.py     #Procesamiento posiciones y movimiento
# ├── mech_analysis.py
# ├── static/
# │   └── js/
# │       └── three_avatar.js      #Codigo Three.js para visualizar el avatar 3d
# └── templates/
#     └── index.html             # Frontend actualizado


# app.py
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
from multicamera_holistic import MultiCameraHolisticBiomechanics
from mech_analysis import BiomechanicalAnalysis

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.active_cameras = set()
        self.biomech = MultiCameraHolisticBiomechanics()
        self.biomech_analysis = BiomechanicalAnalysis()
        self.all_landmarks = {}  # Almacena landmarks de todas las cámaras


    def add_camera(self, camera_id):
        if camera_id not in self.cameras:
            cap = cv2.VideoCapture(camera_id) #r"C:\Users\annim\Downloads\Vídeo sin título (0).mp4" if r"C:\Users\annim\Downloads\Vídeo sin título (0).mp4" else
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #1920
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #1080
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


camera_manager = CameraManager()


@app.route('/')
def index():
    return render_template('index.html')


def generate_frames(camera_id):
    while True:
        frame = camera_manager.get_frame(camera_id)
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('start_camera')
def handle_start_camera(data):
    camera_id = int(data.get('camera_id'))
    if camera_manager.add_camera(camera_id):
        emit('camera_started', {'camera_id': camera_id})


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)