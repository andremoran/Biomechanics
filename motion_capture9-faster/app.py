# /motion_capture/
# Structure:
#├── app.py
#├── multicamera_holistic.py  (tu código original)
#└── templates/
#    └── index.html

#import logging
#logging.basicConfig(level=logging.DEBUG)

#app.py
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import json
import base64
import mediapipe as mp
from io import BytesIO
import threading
import queue
from engineio.async_drivers import threading as async_threading
from multicamera_holistic import MultiCameraHolisticBiomechanics

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', ping_timeout=10)


class OptimizedCameraManager:
    def __init__(self):
        self.cameras = {}
        self.frame_queues = {}
        self.processing_queues = {}
        self.biomech = MultiCameraHolisticBiomechanics()
        self.config = self._configure_performance()

    def _configure_performance(self):
        return {
            'frame_width': 640,
            'frame_height': 480,
            'fps': 30,
            'buffer_size': 1,
            'queue_size': 2
        }

    def add_camera(self, camera_id):
        if camera_id in self.cameras:
            return False

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return False

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame_height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['fps'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['buffer_size'])

        self.cameras[camera_id] = {
            'capture': cap,
            'active': True,
            'holistic': self.biomech.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1,
                enable_segmentation=True,
                refine_face_landmarks=False
            )
        }

        self.frame_queues[camera_id] = queue.Queue(maxsize=self.config['queue_size'])
        self.processing_queues[camera_id] = queue.Queue(maxsize=self.config['queue_size'])

        threading.Thread(target=self._capture_thread, args=(camera_id,), daemon=True).start()
        threading.Thread(target=self._process_thread, args=(camera_id,), daemon=True).start()

        return True

    def _capture_thread(self, camera_id):
        while camera_id in self.cameras and self.cameras[camera_id]['active']:
            cap = self.cameras[camera_id]['capture']
            ret, frame = cap.read()
            if ret:
                if self.frame_queues[camera_id].full():
                    try:
                        self.frame_queues[camera_id].get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queues[camera_id].put(frame)

    def _process_thread(self, camera_id):
        while camera_id in self.cameras and self.cameras[camera_id]['active']:
            try:
                frame = self.frame_queues[camera_id].get(timeout=0.1)
                processed_frame, landmarks = self.biomech.process_frame(
                    frame,
                    self.cameras[camera_id]['holistic']
                )

                if self.processing_queues[camera_id].full():
                    try:
                        self.processing_queues[camera_id].get_nowait()
                    except queue.Empty:
                        pass

                self.processing_queues[camera_id].put((processed_frame, landmarks))

                if landmarks:
                    fig = self.biomech.update_visualization(landmarks)
                    if fig:
                        buf = BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                                    dpi=72, facecolor='black')
                        buf.seek(0)
                        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                        socketio.emit('visualization_update', {'image': plot_data})

            except queue.Empty:
                continue

    def get_frame(self, camera_id):
        if camera_id not in self.cameras or not self.cameras[camera_id]['active']:
            return None, None

        try:
            return self.processing_queues[camera_id].get(timeout=0.1)
        except queue.Empty:
            return None, None

    def remove_camera(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id]['active'] = False
            self.cameras[camera_id]['capture'].release()
            self.cameras[camera_id]['holistic'].close()
            del self.cameras[camera_id]
            del self.frame_queues[camera_id]
            del self.processing_queues[camera_id]


camera_manager = OptimizedCameraManager()


@app.route('/')
def index():
    return render_template('index.html')


def generate_frames(camera_id):
    while True:
        frame, _ = camera_manager.get_frame(camera_id)
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('start_camera')
def handle_start_camera(data):
    camera_id = int(data.get('camera_id'))
    print(f"Starting camera {camera_id}")
    if camera_manager.add_camera(camera_id):
        emit('camera_started', {'camera_id': camera_id})
        print(f"Camera {camera_id} initialized successfully")
    else:
        print(f"Failed to initialize camera {camera_id}")


@socketio.on('stop_camera')
def handle_stop_camera(data):
    camera_id = int(data.get('camera_id'))
    camera_manager.remove_camera(camera_id)
    emit('camera_stopped', {'camera_id': camera_id})


if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)