"""
COMPLETO,FUNCIONA, avanzado, con frames en cada articulación
VERSION1:

Uses pose_0 (basic pose nose point) as the anchor point
Moves the entire detailed face mesh to originate from pose_0
Calculates and applies an offset to maintain the relative positions of all facial features
Preserves the detailed face mesh structure while connecting it to the body's reference point

VERSION2:

Uses face_1 (detailed mesh nose point) as the reference point
Adjusts pose_0 to match the position of face_1
Keeps the detailed face mesh in its original position
Makes the basic pose adapt to the detailed facial tracking"""

import mediapipe as mp
import numpy as np
import cv2
import kineticstoolkit as ktk
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from io import BytesIO

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt


class MultiCameraHolisticBiomechanics:
    def __init__(self, height=1.62, mass=69, face_tracking_version="VERSION1"):
        # Configuración MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Configuración face tracking
        self.face_tracking_version = face_tracking_version
        self.face_reference = {
            'pose_point': 0,
            'mesh_point': 1
        }

        # Inicialización de cámaras y puntos
        self.cameras = {}
        self.exclude_points = list(range(1, 11))
        self.exclude_points.extend([17, 18, 19, 20, 21, 22])

        # Custom body connections
        self.custom_body_connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
            (11, 13), (12, 14),  # Arms
            (23, 25), (24, 26), (25, 27), (26, 28),  # Legs
            (27, 31), (28, 32), (31, 29), (32, 30),  # Feet
        ]



        # Biomechanical parameters
        self.height = height
        self.mass = mass
        self.video_running = False


        # Crear figura estática para visualización web
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initialize KTK visualization
        self.initialize_ktk_components()

    def initialize_ktk_components(self):
        """Initialize KTK visualization con configuración para web"""
        self.points = ktk.TimeSeries(time=np.array([0]))

        # Inicializar puntos (mantener igual)
        for i in range(33):
            if i not in self.exclude_points:
                self.points.data[f"pose_{i}"] = np.array([[0, 0, 0, 1]])

        for hand in ['left', 'right']:
            for i in range(21):
                self.points.data[f"{hand}_hand_{i}"] = np.array([[0, 0, 0, 1]])

        for i in range(468):
            self.points.data[f"face_{i}"] = np.array([[0, 0, 0, 1]])

        # Define enhanced anatomical interconnections
        self.interconnections = {
            "Torso": {
                "Color": [0.2, 0.5, 0.8],
                "Links": [
                    ["pose_11", "pose_12"],  # shoulders
                    ["pose_11", "pose_23"],  # left spine
                    ["pose_12", "pose_24"],  # right spine
                    ["pose_23", "pose_24"],  # hips
                ]
            },
            "ArmL": {
                "Color": [1, 0.25, 0],
                "Links": [
                    ["pose_11", "pose_13"],  # shoulder to elbow
                    ["pose_13", "pose_15"],  # elbow to wrist
                ]
            },
            "ArmR": {
                "Color": [1, 0.25, 0],
                "Links": [
                    ["pose_12", "pose_14"],  # shoulder to elbow
                    ["pose_14", "pose_16"],  # elbow to wrist
                ]
            },
            "HandL": {
                "Color": [1, 0.5, 0],
                "Links": self.generate_detailed_hand_connections("left_hand")
            },
            "HandR": {
                "Color": [1, 0.5, 0],
                "Links": self.generate_detailed_hand_connections("right_hand")
            },
            "Legs": {
                "Color": [0.2, 0.6, 0.8],
                "Links": [
                    ["pose_23", "pose_25"],  # left thigh
                    ["pose_24", "pose_26"],  # right thigh
                    ["pose_25", "pose_27"],  # left calf
                    ["pose_26", "pose_28"],  # right calf
                    ["pose_27", "pose_31"],  # left ankle to foot index
                    ["pose_28", "pose_32"],  # right ankle to foot index
                    ["pose_31", "pose_29"],  # left foot index to heel
                    ["pose_32", "pose_30"],  # right foot index to heel
                    ["pose_27", "pose_29"],  # left ankle index to heel
                    ["pose_28", "pose_30"],  # right ankle index to heel
                ]
            },
            "Face": {
                "Color": [0.2, 0.8, 0.2],
                "Links": self.generate_face_connections()
            }
        }

        # Configurar visualización estática
        self.configure_static_visualization()



    def generate_detailed_hand_connections(self, prefix):
        """Generate anatomically accurate hand connections"""
        connections = []

        # Palm connections (metacarpals)
        palm_points = [0, 1, 5, 9, 13, 17, 0]
        for i in range(len(palm_points) - 1):
            connections.append([
                f"{prefix}_{palm_points[i]}",
                f"{prefix}_{palm_points[i + 1]}"
            ])

        # Finger connections
        finger_starts = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20]
        }

        for finger, points in finger_starts.items():
            for i in range(len(points) - 1):
                connections.append([
                    f"{prefix}_{points[i]}",
                    f"{prefix}_{points[i + 1]}"
                ])

        return connections

    def generate_face_connections(self):
        """Generate comprehensive face mesh connections"""
        connections = []

        # Main facial contour
        contour = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                   397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                   172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        # Eyes
        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 33]
        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 362]

        # Eyebrows
        left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53]
        right_eyebrow = [300, 293, 334, 296, 336, 285, 295, 282, 283]

        # Nose
        nose_bridge = [168, 6, 197, 195, 5, 4]
        nose_bottom = [98, 97, 2, 326, 327]

        # Mouth
        lips_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]
        lips_inner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78]

        # Add all facial feature connections
        feature_sets = [contour, left_eye, right_eye, left_eyebrow, right_eyebrow,
                        nose_bridge, nose_bottom, lips_outer, lips_inner]

        for feature in feature_sets:
            for i in range(len(feature) - 1):
                connections.append([
                    f"face_{feature[i]}",
                    f"face_{feature[i + 1]}"
                ])

            # Close the loop for circular features
            if feature in [left_eye, right_eye, lips_outer, lips_inner]:
                connections.append([
                    f"face_{feature[-1]}",
                    f"face_{feature[0]}"
                ])

        return connections

    def process_frame(self, frame, holistic_instance):
        """Process frame with camera-specific holistic instance"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = holistic_instance.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            self.draw_pose(frame, results)

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                None, self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        return frame, self.process_landmarks(results)

    def draw_pose(self, frame, results):
        """Draw pose landmarks and connections"""
        if results.pose_landmarks:
            # Draw only the custom body connections (excluding simple hands and face)
            for start_idx, end_idx in self.custom_body_connections:
                start_point = results.pose_landmarks.landmark[start_idx]
                end_point = results.pose_landmarks.landmark[end_idx]

                h, w, _ = frame.shape
                start_coord = (int(start_point.x * w), int(start_point.y * h))
                end_coord = (int(end_point.x * w), int(end_point.y * h))

                cv2.line(frame, start_coord, end_coord, (245, 66, 230), 2)

            # Connect elbows to detailed hands
            if results.left_hand_landmarks:
                self.connect_elbow_to_hand(
                    frame,
                    results.pose_landmarks.landmark[13],  # Left elbow
                    results.left_hand_landmarks.landmark[0]  # Left wrist
                )

            if results.right_hand_landmarks:
                self.connect_elbow_to_hand(
                    frame,
                    results.pose_landmarks.landmark[14],  # Right elbow
                    results.right_hand_landmarks.landmark[0]  # Right wrist
                )

    def connect_elbow_to_hand(self, frame, elbow_landmark, wrist_landmark):
        """Draw connection between elbow and detailed hand wrist"""
        h, w, _ = frame.shape
        start_point = (int(elbow_landmark.x * w), int(elbow_landmark.y * h))
        end_point = (int(wrist_landmark.x * w), int(wrist_landmark.y * h))
        cv2.line(frame, start_point, end_point, (245, 66, 230), 2)

    def process_landmarks(self, results):
        """Process landmarks with improved wrist handling"""
        landmarks_3d = {}

        # Process face landmarks first (no changes needed here)
        if results.face_landmarks and results.pose_landmarks:
            if self.face_tracking_version == "VERSION1":
                pose_nose = results.pose_landmarks.landmark[self.face_reference['pose_point']]
                mesh_nose = results.face_landmarks.landmark[self.face_reference['mesh_point']]

                face_offset = {
                    'x': pose_nose.x - mesh_nose.x,
                    'y': pose_nose.y - mesh_nose.y,
                    'z': pose_nose.z - mesh_nose.z
                }

                for idx, landmark in enumerate(results.face_landmarks.landmark):
                    landmarks_3d[f"face_{idx}"] = {
                        'x': landmark.x + face_offset['x'],
                        'y': landmark.y + face_offset['y'],
                        'z': landmark.z + face_offset['z']
                    }

            elif self.face_tracking_version == "VERSION2":
                mesh_nose = results.face_landmarks.landmark[self.face_reference['mesh_point']]

                for idx, landmark in enumerate(results.face_landmarks.landmark):
                    landmarks_3d[f"face_{idx}"] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    }

                if 'pose_0' not in self.exclude_points:
                    landmarks_3d['pose_0'] = {
                        'x': mesh_nose.x,
                        'y': mesh_nose.y,
                        'z': mesh_nose.z
                    }

        # Process body landmarks
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx not in self.exclude_points:
                    if not (self.face_tracking_version == "VERSION2" and
                            idx == 0 and 'pose_0' in landmarks_3d):
                        landmarks_3d[f"pose_{idx}"] = {
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        }

        # Improved hand landmark processing
        if results.left_hand_landmarks and results.pose_landmarks:
            wrist_pose = results.pose_landmarks.landmark[15]  # Left wrist from body

            # Ensure perfect alignment by using the exact same coordinates for hand_0
            landmarks_3d["left_hand_0"] = {
                'x': wrist_pose.x,
                'y': wrist_pose.y,
                'z': wrist_pose.z
            }

            # Calculate hand offset relative to the wrist position
            hand_offset = {
                'x': wrist_pose.x - results.left_hand_landmarks.landmark[0].x,
                'y': wrist_pose.y - results.left_hand_landmarks.landmark[0].y,
                'z': wrist_pose.z - results.left_hand_landmarks.landmark[0].z
            }

            # Apply offset to all hand landmarks except wrist (already set)
            for idx in range(1, 21):
                landmark = results.left_hand_landmarks.landmark[idx]
                landmarks_3d[f"left_hand_{idx}"] = {
                    'x': landmark.x + hand_offset['x'],
                    'y': landmark.y + hand_offset['y'],
                    'z': landmark.z + hand_offset['z']
                }

        if results.right_hand_landmarks and results.pose_landmarks:
            wrist_pose = results.pose_landmarks.landmark[16]  # Right wrist from body

            # Ensure perfect alignment by using the exact same coordinates for hand_0
            landmarks_3d["right_hand_0"] = {
                'x': wrist_pose.x,
                'y': wrist_pose.y,
                'z': wrist_pose.z
            }

            # Calculate hand offset relative to the wrist position
            hand_offset = {
                'x': wrist_pose.x - results.right_hand_landmarks.landmark[0].x,
                'y': wrist_pose.y - results.right_hand_landmarks.landmark[0].y,
                'z': wrist_pose.z - results.right_hand_landmarks.landmark[0].z
            }

            # Apply offset to all hand landmarks except wrist (already set)
            for idx in range(1, 21):
                landmark = results.right_hand_landmarks.landmark[idx]
                landmarks_3d[f"right_hand_{idx}"] = {
                    'x': landmark.x + hand_offset['x'],
                    'y': landmark.y + hand_offset['y'],
                    'z': landmark.z + hand_offset['z']
                }

        return landmarks_3d

    def calculate_arm_angles(self, landmarks_3d, scale_factor, reference_point):
        """
        Calculate joint angles for arms and legs with anatomical frames
        """
        angles = {}

        # Define segment pairs for angle calculations
        segments = {
            'left_arm': {
                'shoulder': ('pose_11', 'pose_13'),
                'elbow': ('pose_13', 'pose_15'),
                'wrist': ('pose_15', 'left_hand_1')
            },
            'right_arm': {
                'shoulder': ('pose_12', 'pose_14'),
                'elbow': ('pose_14', 'pose_16'),
                'wrist': ('pose_16', 'right_hand_1')
            },
            'left_leg': {
                'hip': ('pose_23', 'pose_25'),
                'knee': ('pose_25', 'pose_27'),
                'ankle': ('pose_27', 'pose_31')
            },
            'right_leg': {
                'hip': ('pose_24', 'pose_26'),
                'knee': ('pose_26', 'pose_28'),
                'ankle': ('pose_28', 'pose_32')
            }
        }

        def create_anatomical_frame(prox_point, dist_point, prev_z=None):
            """Helper function to create anatomical frames"""
            y = dist_point - prox_point
            y = y / np.linalg.norm(y)

            if prev_z is None:
                temp = np.array([0, 0, 1])
            else:
                temp = prev_z

            x = np.cross(y, temp)
            if np.linalg.norm(x) < 1e-10:
                temp = np.array([1, 0, 0])
                x = np.cross(y, temp)
            x = x / np.linalg.norm(x)
            z = np.cross(x, y)

            return x, y, z

        for limb, joints in segments.items():
            prev_z = None

            for joint, (prox, dist) in joints.items():
                if prox in landmarks_3d and dist in landmarks_3d:
                    # Get scaled coordinates
                    prox_point = np.array([
                        (landmarks_3d[prox]['x'] - reference_point['x']) * scale_factor,
                        (landmarks_3d[prox]['y'] - reference_point['y']) * scale_factor,
                        (landmarks_3d[prox]['z'] - reference_point['z']) * scale_factor
                    ])

                    dist_point = np.array([
                        (landmarks_3d[dist]['x'] - reference_point['x']) * scale_factor,
                        (landmarks_3d[dist]['y'] - reference_point['y']) * scale_factor,
                        (landmarks_3d[dist]['z'] - reference_point['z']) * scale_factor
                    ])

                    # Create anatomical frame
                    x, y, z = create_anatomical_frame(prox_point, dist_point, prev_z)
                    prev_z = z

                    # Create transformation matrix
                    rotation_matrix = np.eye(4)
                    rotation_matrix[:3, 0] = x
                    rotation_matrix[:3, 1] = y
                    rotation_matrix[:3, 2] = z
                    rotation_matrix[:3, 3] = prox_point

                    # Store frame
                    frame_name = f"{limb}_{joint}_frame"
                    self.points.data[frame_name] = np.array([rotation_matrix])

                    # Calculate angles for non-root joints
                    if joint not in ['shoulder', 'hip']:
                        prev_joint = list(joints.keys())[list(joints.keys()).index(joint) - 1]
                        prev_frame_name = f"{limb}_{prev_joint}_frame"

                        if prev_frame_name in self.points.data:
                            prev_frame = self.points.data[prev_frame_name][0]
                            relative_transform = ktk.geometry.get_local_coordinates(
                                np.array([rotation_matrix]),
                                np.array([prev_frame])
                            )
                            joint_angles = ktk.geometry.get_angles(
                                relative_transform,
                                seq="ZXY",
                                degrees=True
                            )[0]

                            angles[f"{limb}_{joint}"] = {
                                'flexion': joint_angles[0],
                                'carrying_angle': joint_angles[1],
                                'rotation': joint_angles[2]
                            }

        return angles

    def configure_static_visualization(self):
        """Configurar visualización estática para web"""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.view_init(elev=15, azim=45)

    def configure_static_visualization(self):
        """Configurar visualización estática para web"""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_box_aspect([1, 1, 1])

        # Ajustar el ángulo de vista para ver el avatar de frente en el plano X-Y
        self.ax.view_init(azim=-45)

        # Configurar fondo negro y elementos en blanco
        self.ax.set_facecolor('black')
        # Set all panes and grid lines to black
        self.ax.xaxis.set_pane_color((0, 0, 0, 1))
        self.ax.yaxis.set_pane_color((0, 0, 0, 1))
        self.ax.zaxis.set_pane_color((0, 0, 0, 1))

        # Make grid lines black
        self.ax.grid(False)

    def update_visualization(self, landmarks_3d):
        """Versión modificada para web con visualización dinámica y reorientación"""
        if not landmarks_3d:
            return None

        self.ax.clear()
        all_points = []

        for landmark_id, coords in landmarks_3d.items():
            if landmark_id in self.points.data:
                # Reorientar coordenadas: Y↔Z para que el avatar esté en el plano X-Y
                scaled_coords = np.array([
                    coords['x'] * self.height,
                    coords['z'] * self.height,  # Z pasa a Y
                    -coords['y'] * self.height,  # Y pasa a Z (negativo para orientación correcta)
                    1
                ])
                self.points.data[landmark_id] = np.array([scaled_coords])
                all_points.append(scaled_coords[:3])

        all_points = np.array(all_points)

        if len(all_points) > 0:
            min_coords = np.min(all_points, axis=0)
            max_coords = np.max(all_points, axis=0)
            center = (min_coords + max_coords) / 2
            range_size = max_coords - min_coords
            max_range = np.max(range_size)
            margin = max_range * 0.1

            for i, (label, axis) in enumerate(
                    zip(['x', 'y', 'z'], [self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim])):
                axis([center[i] - max_range / 2 - margin, center[i] + max_range / 2 + margin])

        for segment_name, segment_data in self.interconnections.items():
            for link in segment_data["Links"]:
                if link[0] in self.points.data and link[1] in self.points.data:
                    point1 = self.points.data[link[0]][0][:3]
                    point2 = self.points.data[link[1]][0][:3]
                    self.ax.plot([point1[0], point2[0]],
                                 [point1[1], point2[1]],
                                 [point1[2], point2[2]],
                                 color=segment_data["Color"],
                                 linewidth=2)

        self.configure_static_visualization()
        self.fig.set_size_inches(12, 10)
        self.fig.tight_layout()

        return self.fig

    def enable_camera(self, camera_index=0):
        """Versión modificada para manejo de cámaras web"""
        if camera_index not in self.cameras:
            self.cameras[camera_index] = {
                'holistic': self.mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=2,
                    enable_segmentation=True,
                    refine_face_landmarks=True
                )
            }
            return True
        return False

    def run(self):
        """Main processing loop with optimized camera handling"""
        while self.video_running:
            all_landmarks_3d = {}
            active_cameras = dict(self.cameras)  # Create copy to avoid runtime modification issues

            for camera_idx, camera_data in active_cameras.items():
                ret, frame = camera_data['capture'].read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                processed_frame, frame_landmarks = self.process_frame(
                    frame, camera_data['holistic']
                )

                if frame_landmarks:
                    camera_data['landmarks_3d'] = frame_landmarks
                    all_landmarks_3d.update(frame_landmarks)

                cv2.imshow(f'Camera {camera_idx}', processed_frame)

            if all_landmarks_3d:
                self.update_visualization(all_landmarks_3d)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def merge_landmarks(self, all_landmarks, new_landmarks):
        """Merge landmarks from multiple cameras"""
        all_landmarks.update(new_landmarks)  # Simple update is sufficient for most cases

    def cleanup(self):
        """Limpieza modificada para entorno web"""
        for camera in self.cameras.values():
            if 'holistic' in camera:
                camera['holistic'].close()
        self.cameras.clear()
        plt.close(self.fig)


def main():
    capture = MultiCameraHolisticBiomechanics(face_tracking_version="VERSION1")
    capture.enable_camera(0)
    capture.enable_camera(1)
    capture.enable_camera(2)
    capture.run()


if __name__ == "__main__":
    main()