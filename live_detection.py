# """
# Live Face Detection Module - CPU Version
# Integrates with Flask app for real-time face detection, age/gender detection, and quality filtering.
# """
# import cv2
# import numpy as np
# from datetime import datetime
# from pathlib import Path
# import time
# import threading
# import queue
# import os
# from typing import Dict, List, Optional, Tuple
# from collections import defaultdict
# import math

# # ============================================================================
# # CONFIGURATION - CPU VERSION
# # ============================================================================
# class LiveDetectionConfig:
#     # Camera settings (Tiandy)
#     CAMERA_IP = "192.168.25.13"
#     CAMERA_USERNAME = "admin"
#     CAMERA_PASSWORD = "Pass@123"
#     CAMERA_PORT = 554
    
#     STREAM_PATHS = [
#         "Streaming/Channels/101",
#         "Streaming/Channels/102", 
#         "Streaming/Channels/1",
#         "stream1",
#         "stream2",
#         "live",
#         "h264",
#         "cam/realmonitor?channel=1&subtype=0"
#     ]
    
#     # Detection settings
#     FACE_CROP_SIZE = 160  # REID standard size
#     CROP_PADDING = 20
#     COOLDOWN_SECONDS = 3.0
#     JPEG_QUALITY = 95
    
#     # Quality filter settings
#     QUALITY_MIN_SCORE = 0.6
#     QUALITY_MIN_FACE_SIZE = 80
#     QUALITY_MAX_ANGLE = 15.0  # Strict: 15 degrees for frontal
#     QUALITY_MIN_CONFIDENCE = 0.7
    
#     # Speed optimization
#     DETECTION_INTERVAL = 1
#     DISPLAY_INTERVAL = 1
#     FRAME_SKIP = 3
    
#     # RTSP connection
#     RTSP_TIMEOUT = 5
#     RTSP_RETRY_COUNT = 3
#     RTSP_BUFFER_SIZE = 1

# # ============================================================================
# # QUALITY FILTER (SIMPLIFIED - NO LANDMARKS REQUIRED)
# # ============================================================================
# class QualityFilter:
#     """Quality filter to save only the best face crop per person."""
    
#     def __init__(
#         self,
#         min_quality_score: float = 0.6,
#         min_face_size: int = 80,
#         max_angle_threshold: float = 15.0,
#         min_confidence: float = 0.7
#     ):
#         self.min_quality_score = min_quality_score
#         self.min_face_size = min_face_size
#         self.max_angle_threshold = max_angle_threshold
#         self.min_confidence = min_confidence
        
#         self.best_faces: Dict[int, Dict] = {}
#         self.quality_history: Dict[int, List[float]] = {}
#         self.history_window = 5
#         self.saved_persons: set = set()
        
#         # Statistics
#         self.total_evaluated = 0
#         self.total_saved = 0
#         self.total_rejected = 0
    
#     def calculate_quality_metrics(self, detection: Dict) -> Dict:
#         """Calculate comprehensive quality metrics for a face detection."""
#         bbox = detection.get('bbox', [0, 0, 0, 0])
#         quality_score = detection.get('quality', detection.get('score', 0.0))
#         confidence = detection.get('score', 0.0)
        
#         x1, y1, x2, y2 = bbox
#         width = x2 - x1
#         height = y2 - y1
#         size = max(width, height)
        
#         # Estimate angle from bbox aspect ratio (simplified)
#         aspect_ratio = width / height if height > 0 else 1.0
#         angle_estimate = abs(1.0 - aspect_ratio) * 30.0
#         max_angle = min(angle_estimate, 15.0)
        
#         angle_score = max(0.0, 1.0 - (max_angle / self.max_angle_threshold))
#         size_score = min(1.0, size / 150.0)
        
#         combined_quality = (
#             quality_score * 0.4 +
#             angle_score * 0.3 +
#             size_score * 0.2 +
#             confidence * 0.1
#         )
        
#         is_valid = (
#             combined_quality >= self.min_quality_score and
#             size >= self.min_face_size and
#             confidence >= self.min_confidence and
#             max_angle < self.max_angle_threshold
#         )
        
#         return {
#             'combined_quality': combined_quality,
#             'angle_score': angle_score,
#             'size_score': size_score,
#             'max_angle': max_angle,
#             'face_size': size,
#             'quality_score': quality_score,
#             'confidence': confidence,
#             'is_frontal': max_angle < 15.0,
#             'is_valid': is_valid
#         }
    
#     def should_save(self, detection: Dict, track_id: int) -> Tuple[bool, Dict]:
#         """Determine if this face should be saved."""
#         self.total_evaluated += 1
#         metrics = self.calculate_quality_metrics(detection)
        
#         # Perfect face: always save
#         is_perfect_face = (
#             metrics.get('is_valid', False) and
#             metrics.get('combined_quality', 0) > 0.75 and
#             metrics.get('max_angle', 999) < 10.0
#         )
        
#         if is_perfect_face:
#             current_quality = metrics['combined_quality']
#             if track_id not in self.quality_history:
#                 self.quality_history[track_id] = []
#             self.quality_history[track_id].append(current_quality)
#             if len(self.quality_history[track_id]) > self.history_window:
#                 self.quality_history[track_id] = self.quality_history[track_id][-self.history_window:]
            
#             self.best_faces[track_id] = {
#                 'detection': detection,
#                 'metrics': metrics,
#                 'combined_quality': current_quality
#             }
#             self.saved_persons.add(track_id)
#             self.total_saved += 1
#             return (True, metrics)
        
#         # One person = one snapshot
#         if track_id in self.saved_persons:
#             self.total_rejected += 1
#             return (False, {'rejected': 'Person already saved'})
        
#         if not metrics['is_valid']:
#             self.total_rejected += 1
#             return (False, metrics)
        
#         if track_id not in self.quality_history:
#             self.quality_history[track_id] = []
        
#         current_quality = metrics['combined_quality']
#         self.quality_history[track_id].append(current_quality)
        
#         if len(self.quality_history[track_id]) > self.history_window:
#             self.quality_history[track_id] = self.quality_history[track_id][-self.history_window:]
        
#         is_local_maximum = self._is_quality_local_maximum(track_id, current_quality)
        
#         if track_id not in self.best_faces:
#             self.best_faces[track_id] = {
#                 'detection': detection,
#                 'metrics': metrics,
#                 'combined_quality': current_quality
#             }
#             self.saved_persons.add(track_id)
#             self.total_saved += 1
#             return (True, metrics)
        
#         current_best = self.best_faces[track_id]
#         if current_quality > current_best['combined_quality']:
#             self.best_faces[track_id] = {
#                 'detection': detection,
#                 'metrics': metrics,
#                 'combined_quality': current_quality
#             }
        
#         if current_quality >= current_best['combined_quality'] and is_local_maximum:
#             self.saved_persons.add(track_id)
#             self.total_saved += 1
#             return (True, metrics)
#         else:
#             self.total_rejected += 1
#             return (False, metrics)
    
#     def _is_quality_local_maximum(self, track_id: int, current_quality: float) -> bool:
#         """Check if current quality is a local maximum."""
#         history = self.quality_history.get(track_id, [])
#         if len(history) < 3:
#             return False
        
#         current_idx = len(history) - 1
#         if current_idx > 0:
#             if current_quality < history[current_idx - 1]:
#                 return False
        
#         if len(history) >= self.history_window:
#             recent_max = max(history[-self.history_window:])
#             if current_quality >= recent_max * 0.98:
#                 return True
        
#         return False
    
#     def get_statistics(self) -> Dict:
#         """Get filter statistics."""
#         return {
#             'total_evaluated': self.total_evaluated,
#             'total_saved': self.total_saved,
#             'total_rejected': self.total_rejected,
#             'active_tracks': len(self.best_faces),
#             'saving_rate': self.total_saved / max(1, self.total_evaluated)
#         }

# # ============================================================================
# # TIANDY CAMERA CONNECTION
# # ============================================================================
# class TiandyCamera:
#     def __init__(self):
#         self.cap = None
    
#     def build_rtsp_url(self, stream_path):
#         return (
#             f"rtsp://{LiveDetectionConfig.CAMERA_USERNAME}:{LiveDetectionConfig.CAMERA_PASSWORD}"
#             f"@{LiveDetectionConfig.CAMERA_IP}:{LiveDetectionConfig.CAMERA_PORT}/{stream_path}"
#         )
    
#     def connect(self):
#         print("=" * 70)
#         print("CONNECTING TO TIANDY CAMERA (CPU Version)")
#         print("=" * 70)
#         print(f"Camera IP: {LiveDetectionConfig.CAMERA_IP}")
        
#         for stream_path in LiveDetectionConfig.STREAM_PATHS:
#             url = self.build_rtsp_url(stream_path)
#             print(f"Trying RTSP -> {url}")
            
#             for attempt in range(LiveDetectionConfig.RTSP_RETRY_COUNT):
#                 try:
#                     self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#                     self.cap.set(cv2.CAP_PROP_BUFFERSIZE, LiveDetectionConfig.RTSP_BUFFER_SIZE)
                    
#                     start_time = time.time()
#                     while not self.cap.isOpened() and (time.time() - start_time) < LiveDetectionConfig.RTSP_TIMEOUT:
#                         time.sleep(0.1)
                    
#                     if self.cap.isOpened():
#                         ret, frame = self.cap.read()
#                         if ret and frame is not None and frame.size > 0:
#                             print(f"✓ CONNECTED via FFMPEG (attempt {attempt + 1})")
#                             print(f"  Frame size: {frame.shape[1]}x{frame.shape[0]}")
#                             return True
#                         else:
#                             self.cap.release()
#                             self.cap = None
#                 except Exception as e:
#                     print(f"  Connection attempt {attempt + 1} failed: {e}")
#                     if self.cap:
#                         self.cap.release()
#                         self.cap = None
#                     time.sleep(1)
        
#         print("\n✗ FAILED to connect to camera!")
#         return False
    
#     def read(self):
#         if not self.cap or not self.cap.isOpened():
#             return False, None
#         return self.cap.read()
    
#     def release(self):
#         if self.cap:
#             self.cap.release()

# # ============================================================================
# # LIVE DETECTION MANAGER
# # ============================================================================
# class LiveDetectionManager:
#     """Manages live face detection process."""
    
#     def __init__(self, face_detection_func, save_callback, cropped_dir):
#         self.face_detection_func = face_detection_func  # Function from cla.py
#         self.save_callback = save_callback  # Callback to save faces
#         self.cropped_dir = Path(cropped_dir)
#         self.cropped_dir.mkdir(exist_ok=True, parents=True)
        
#         self.camera = TiandyCamera()
#         self.quality_filter = QualityFilter(
#             min_quality_score=LiveDetectionConfig.QUALITY_MIN_SCORE,
#             min_face_size=LiveDetectionConfig.QUALITY_MIN_FACE_SIZE,
#             max_angle_threshold=LiveDetectionConfig.QUALITY_MAX_ANGLE,
#             min_confidence=LiveDetectionConfig.QUALITY_MIN_CONFIDENCE
#         )
        
#         self.is_running = False
#         self.exit_event = threading.Event()
#         self.frame_queue = queue.Queue(maxsize=10)
#         self.detection_thread = None
#         self.capture_thread = None
        
#         # Latest results for API
#         self.latest_frame = None
#         self.latest_detections = []
#         self.lock = threading.Lock()
        
#         # Statistics
#         self.face_counter = 0
#         self.last_save_time = {}
    
#     def start(self):
#         """Start live detection."""
#         if self.is_running:
#             return False
        
#         if not self.camera.connect():
#             return False
        
#         self.is_running = True
#         self.exit_event.clear()
        
#         self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
#         self.detection_thread = threading.Thread(target=self._process_frames, daemon=True)
        
#         self.capture_thread.start()
#         self.detection_thread.start()
        
#         return True
    
#     def stop(self):
#         """Stop live detection."""
#         if not self.is_running:
#             return False
        
#         self.is_running = False
#         self.exit_event.set()
        
#         if self.capture_thread:
#             self.capture_thread.join(timeout=2)
#         if self.detection_thread:
#             self.detection_thread.join(timeout=2)
        
#         self.camera.release()
#         return True
    
#     def _capture_frames(self):
#         """Capture frames from camera."""
#         consecutive_errors = 0
#         max_errors = 30
        
#         while not self.exit_event.is_set():
#             ret, frame = self.camera.read()
            
#             if ret and frame is not None:
#                 consecutive_errors = 0
#                 if self.frame_queue.full():
#                     try:
#                         self.frame_queue.get_nowait()
#                     except queue.Empty:
#                         pass
                
#                 try:
#                     self.frame_queue.put_nowait(frame)
#                 except queue.Full:
#                     pass
#             else:
#                 consecutive_errors += 1
#                 if consecutive_errors >= max_errors:
#                     print(f"[Warning] Camera read errors: {consecutive_errors}")
#                     time.sleep(1)
#                     consecutive_errors = 0
            
#             time.sleep(0.005)
    
#     def _process_frames(self):
#         """Process frames for face detection."""
#         frame_count = 0
        
#         while not self.exit_event.is_set():
#             try:
#                 frame = self.frame_queue.get(timeout=0.05)
#             except queue.Empty:
#                 time.sleep(0.001)
#                 continue
            
#             frame_count += 1
            
#             # Convert BGR to RGB for face detection
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Detect faces using existing function
#             if frame_count % LiveDetectionConfig.DETECTION_INTERVAL == 0:
#                 detected_faces = self.face_detection_func(frame_rgb, skip_large_arrays=True)
                
#                 # Update latest results
#                 with self.lock:
#                     self.latest_frame = frame.copy()
#                     self.latest_detections = []
                    
#                     for i, face in enumerate(detected_faces):
#                         bbox = face.get('bbox', [])
#                         if len(bbox) >= 4:
#                             x1, y1, x2, y2 = bbox[:4]
                            
#                             # Check quality
#                             track_id = frame_count * 1000 + i
#                             # Use detection confidence or default quality
#                             confidence = 0.8
#                             if 'confidence' in face:
#                                 conf_dict = face['confidence']
#                                 if isinstance(conf_dict, dict):
#                                     confidence = max(conf_dict.get('gender', 0.8), conf_dict.get('age', 0.8))
                            
#                             detection_data = {
#                                 'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                                 'score': confidence,
#                                 'quality': confidence
#                             }
                            
#                             should_save, quality_metrics = self.quality_filter.should_save(
#                                 detection_data, track_id
#                             )
                            
#                             # Extract face crop
#                             h, w = frame.shape[:2]
#                             x1_int = max(0, int(x1))
#                             y1_int = max(0, int(y1))
#                             x2_int = min(w, int(x2))
#                             y2_int = min(h, int(y2))
                            
#                             if x2_int > x1_int and y2_int > y1_int:
#                                 face_crop = frame[y1_int:y2_int, x1_int:x2_int]
                                
#                                 # Save if quality is good
#                                 if should_save and self._can_save(track_id):
#                                     self._save_face(face_crop, face, quality_metrics, track_id)
                                
#                                 # Add to detections
#                                 quality_score = quality_metrics.get('combined_quality', 0.0)
#                                 self.latest_detections.append({
#                                     'bbox': [x1_int, y1_int, x2_int, y2_int],
#                                     'age': face.get('age', 'Unknown'),
#                                     'gender': face.get('gender', 'Unknown'),
#                                     'emotion': face.get('emotion', 'Neutral'),
#                                     'person_id': face.get('person_id', ''),
#                                     'quality_score': quality_score,
#                                     'quality': quality_score > 0.7
#                                 })
                
#     def _can_save(self, track_id):
#         """Check if can save (cooldown check)."""
#         if track_id not in self.last_save_time:
#             return True
#         return (time.time() - self.last_save_time[track_id]) > LiveDetectionConfig.COOLDOWN_SECONDS
    
#     def _save_face(self, face_crop, face_data, quality_metrics, track_id):
#         """Save face crop to disk."""
#         try:
#             # Resize to standard size
#             if face_crop.shape[0] != LiveDetectionConfig.FACE_CROP_SIZE or \
#                face_crop.shape[1] != LiveDetectionConfig.FACE_CROP_SIZE:
#                 if face_crop.shape[0] < LiveDetectionConfig.FACE_CROP_SIZE:
#                     face_crop = cv2.resize(face_crop, 
#                                          (LiveDetectionConfig.FACE_CROP_SIZE, LiveDetectionConfig.FACE_CROP_SIZE),
#                                          interpolation=cv2.INTER_CUBIC)
#                 else:
#                     face_crop = cv2.resize(face_crop,
#                                          (LiveDetectionConfig.FACE_CROP_SIZE, LiveDetectionConfig.FACE_CROP_SIZE),
#                                          interpolation=cv2.INTER_AREA)
            
#             self.face_counter += 1
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             quality_str = f"_q{int(quality_metrics.get('combined_quality', 0) * 100):03d}"
#             filename = f"Face_{self.face_counter}_{timestamp}{quality_str}.jpg"
#             filepath = self.cropped_dir / filename
            
#             cv2.imwrite(str(filepath), face_crop, 
#                        [cv2.IMWRITE_JPEG_QUALITY, LiveDetectionConfig.JPEG_QUALITY])
            
#             self.last_save_time[track_id] = time.time()
            
#             print(f"[LIVE SAVED] {filename} - Quality: {quality_metrics.get('combined_quality', 0):.2f}, "
#                   f"Gender: {face_data.get('gender', 'Unknown')}, Age: {face_data.get('age', 'Unknown')}")
            
#         except Exception as e:
#             print(f"[LIVE SAVE ERROR] {e}")
    
#     def get_latest_frame(self):
#         """Get latest frame (thread-safe)."""
#         with self.lock:
#             return self.latest_frame.copy() if self.latest_frame is not None else None
    
#     def get_latest_detections(self):
#         """Get latest detections (thread-safe)."""
#         with self.lock:
#             return self.latest_detections.copy()
    
#     def get_status(self):
#         """Get detection status."""
#         stats = self.quality_filter.get_statistics()
#         return {
#             'is_running': self.is_running,
#             'faces_detected': len(self.latest_detections),
#             'faces_saved': self.face_counter,
#             'statistics': stats
#         }

import cv2
import numpy as np
import os
import time
import threading
from collections import defaultdict
from insightface import app

CAMERA_IP = "192.168.25.13"
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "Pass@123"
CAMERA_PORT = 554

STREAM_PATHS = [
    "Streaming/Channels/101",
    "Streaming/Channels/102", 
    "Streaming/Channels/1",
    "stream1",
    "stream2",
    "live",
    "h264",
    "cam/realmonitor?channel=1&subtype=0"
]

class FaceSnapSystem:
    _person_id_lock = threading.Lock()
    _shared_person_id = None
    _gui_available = None
    
    def __init__(self, rtsp_url, camera_name="Camera"):
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name
        self.faceapp = app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.faceapp.prepare(ctx_id=-1, det_size=(640, 640))
        self.snapshot_dir =  r"C:\Users\USER\Downloads\newcla\snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.tracked_persons = defaultdict(lambda: {'last_snap': 0, 'snapped': False})
        self.cooldown_seconds = 20
        self.grid_size = 50
        
        if FaceSnapSystem._shared_person_id is None:
            FaceSnapSystem._shared_person_id = self._get_next_person_id()
        
        if FaceSnapSystem._gui_available is None:
            FaceSnapSystem._gui_available = self._check_gui_available()
        
    def _check_gui_available(self):
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('test', test_img)
            cv2.destroyAllWindows()
            return True
        except:
            return False
    
    def _get_next_person_id(self):
        if os.path.exists('database/last_person_id.txt'):
            with open('database/last_person_id.txt', 'r') as f:
                return int(f.read().strip()) + 1
        os.makedirs('database', exist_ok=True)
        return 1
    
    def _get_and_increment_person_id(self):
        with FaceSnapSystem._person_id_lock:
            person_id = FaceSnapSystem._shared_person_id
            FaceSnapSystem._shared_person_id += 1
            self._save_person_id(FaceSnapSystem._shared_person_id)
            return person_id
    
    def _save_person_id(self, person_id):
        os.makedirs('database', exist_ok=True)
        with open('database/last_person_id.txt', 'w') as f:
            f.write(str(person_id))
    
    def _get_grid_key(self, bbox):
        center_x = int((bbox[0] + bbox[2]) / 2 / self.grid_size)
        center_y = int((bbox[1] + bbox[3]) / 2 / self.grid_size)
        return (center_x, center_y)
    
    def _estimate_distance(self, face_width_pixels):
        reference_width_mm = 140
        reference_distance_m = 3.0
        reference_width_pixels = 120
        if face_width_pixels > 0:
            distance_m = (reference_distance_m * reference_width_pixels) / face_width_pixels
            return distance_m
        return 0.0
    
    def _estimate_pose(self, landmarks):
        if len(landmarks) < 5:
            return None, None
        
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        
        eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        nose_offset_x = nose[0] - eye_center_x
        yaw_deg = np.degrees(np.arcsin(np.clip(nose_offset_x / (eye_distance / 2 + 1e-6), -1, 1)))
        
        face_vertical = np.sqrt((mouth_center_x - eye_center_x)**2 + (mouth_center_y - eye_center_y)**2)
        nose_offset_y = nose[1] - eye_center_y
        pitch_deg = np.degrees(np.arcsin(np.clip(nose_offset_y / (face_vertical + 1e-6), -1, 1)))
        
        return yaw_deg, pitch_deg
    
    def _check_face_acceptance(self, yaw, pitch):
        if yaw is None or pitch is None:
            return False, "No pose"
        if abs(yaw) > 5:
            return False, "Side face"
        if abs(pitch) > 10:
            return False, "Looking up/down"
        return True, "Frontal"
    
    def _has_valid_landmarks(self, face):
        if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
            return False
        landmarks = face.landmark_2d_106
        if landmarks.shape[0] < 106:
            return False
        return True
    
    def _get_5_landmarks(self, face):
        landmarks = face.landmark_2d_106
        left_eye_idx = 74
        right_eye_idx = 77
        nose_idx = 86
        left_mouth_idx = 84
        right_mouth_idx = 90
        
        left_eye = landmarks[left_eye_idx]
        right_eye = landmarks[right_eye_idx]
        nose = landmarks[nose_idx]
        left_mouth = landmarks[left_mouth_idx]
        right_mouth = landmarks[right_mouth_idx]
        
        return [
            [left_eye[0], left_eye[1]],
            [right_eye[0], right_eye[1]],
            [nose[0], nose[1]],
            [left_mouth[0], left_mouth[1]],
            [right_mouth[0], right_mouth[1]]
        ]
    
    def _crop_face(self, frame, bbox, landmarks):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        w = x2 - x1
        h = y2 - y1
        size = max(w, h)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        crop_x1 = max(0, center_x - size // 2)
        crop_y1 = max(0, center_y - size // 2)
        crop_x2 = min(frame.shape[1], crop_x1 + size)
        crop_y2 = min(frame.shape[0], crop_y1 + size)
        
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        resized = cv2.resize(cropped, (160, 160))
        return resized
    
    def _get_age_gender(self, face):
        try:
            if hasattr(face, 'age') and face.age is not None:
                age = int(face.age)
                gender = 'M' if (hasattr(face, 'gender') and face.gender == 1) else 'F'
                return age, gender
        except:
            pass
        return None, None
    
    def _snap_face(self, frame, bbox, landmarks):
        current_time = time.time()
        grid_key = self._get_grid_key(bbox)
        
        if grid_key in self.tracked_persons:
            person_data = self.tracked_persons[grid_key]
            if person_data['snapped']:
                if current_time - person_data['last_snap'] < self.cooldown_seconds:
                    return False
                return False
        
        person_id = self._get_and_increment_person_id()
        
        person_folder = os.path.join(self.snapshot_dir, f'Person_{person_id}')
        os.makedirs(person_folder, exist_ok=True)
        
        cropped_face = self._crop_face(frame, bbox, landmarks)
        filename = os.path.join(person_folder, f'face_{int(time.time())}.jpg')
        cv2.imwrite(filename, cropped_face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        self.tracked_persons[grid_key]['last_snap'] = current_time
        self.tracked_persons[grid_key]['snapped'] = True
        self.tracked_persons[grid_key]['person_id'] = person_id
        
        return True
    
    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print(f"[{self.camera_name}] Failed to open RTSP stream: {self.rtsp_url}")
            return
        
        print(f"[{self.camera_name}] Connected successfully")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            faces = self.faceapp.get(frame)
            
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                face_width = x2 - x1
                distance = self._estimate_distance(face_width)
                
                age, gender = self._get_age_gender(face)
                
                if not self._has_valid_landmarks(face):
                    color = (0, 0, 255)
                    status = "No landmarks"
                else:
                    landmarks_5 = self._get_5_landmarks(face)
                    yaw, pitch = self._estimate_pose(landmarks_5)
                    accepted, reason = self._check_face_acceptance(yaw, pitch)
                    
                    if accepted and 3 <= distance <= 5:
                        color = (0, 255, 0)
                        status = "ACCEPT"
                        if self._snap_face(frame, bbox, landmarks_5):
                            status = "SNAPPED"
                    else:
                        color = (0, 0, 255)
                        if distance < 3:
                            status = "Too close"
                        elif distance > 5:
                            status = "Too far"
                        else:
                            status = reason
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                info_text = f"Dist: {distance:.1f}m"
                if age is not None and gender is not None:
                    info_text += f" | {age} {gender}"
                info_text += f" | {status}"
                
                cv2.putText(frame, info_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if FaceSnapSystem._gui_available:
                cv2.putText(frame, self.camera_name, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                try:
                    cv2.imshow(f'Face Snap System - {self.camera_name}', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    FaceSnapSystem._gui_available = False
            else:
                time.sleep(0.03)
        
        cap.release()
        if FaceSnapSystem._gui_available:
            try:
                cv2.destroyAllWindows()
            except:
                pass

def build_rtsp_url(ip, username, password, port, stream_path):
    return f"rtsp://{username}:{password}@{ip}:{port}/{stream_path}"

if __name__ == '__main__':
    working_stream = None
    tested_paths = []
    
    priority_paths = ["Streaming/Channels/101", "Streaming/Channels/102"]
    for stream_path in priority_paths + [p for p in STREAM_PATHS if p not in priority_paths]:
        if stream_path in tested_paths:
            continue
        tested_paths.append(stream_path)
        rtsp_url = build_rtsp_url(CAMERA_IP, CAMERA_USERNAME, CAMERA_PASSWORD, CAMERA_PORT, stream_path)
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                working_stream = (rtsp_url, stream_path)
                break
    
    if not working_stream:
        print("No working stream found")
        exit(1)
    
    print(f"Camera stream: {working_stream[1]} -> {working_stream[0]}")
    
    system = FaceSnapSystem(working_stream[0], "Camera_1")
    try:
        system.run()
    except KeyboardInterrupt:
        print("Shutting down...")
        cv2.destroyAllWindows()

