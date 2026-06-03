#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Application Đơn Giản cho Hệ Thống Đếm Phương Tiện
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import sqlite3
from datetime import datetime
import threading
import time
import base64
from collections import defaultdict, deque

app = Flask(__name__)
app.secret_key = 'traffic_counter_2024'

# Global variables
current_counter = None
is_processing = False

class SimpleTrafficCounter:
    def __init__(self, model_path='traffic_training_small/traffic_yolov8n_small/weights/best.pt'):
        try:
            self.model = YOLO(model_path)
        except:
            # Fallback to default model
            self.model = YOLO('yolov8n.pt')
        
        # Variables
        self.vehicles = {}
        self.next_id = 0
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        
        # Line
        self.line_y = None
        self.line_start = None
        self.line_end = None
        
        # Counts
        self.total_count = 0
        self.current_session_count = 0
        self.total_detected = 0
        self.total_crossed = 0
        
        # Performance
        self.detection_interval = 0.2
        self.last_detection = 0
        
        # Video stream
        self.latest_frame = None
        
    def set_line_position(self, frame_height, frame_width):
        if self.line_y is None:
            self.line_y = frame_height // 2
            self.line_start = (50, self.line_y)
            self.line_end = (frame_width - 50, self.line_y)
    
    def set_custom_line(self, start_point, end_point):
        self.line_start = start_point
        self.line_end = end_point
        self.line_y = (start_point[1] + end_point[1]) // 2
    
    def check_crossing(self, vehicle):
        if self.line_y is None or vehicle.id not in self.track_history:
            return False
        
        points = list(self.track_history[vehicle.id])
        if len(points) < 3:
            return False
        
        tolerance = 10
        
        if len(points) >= 3:
            point1_y = points[-3][1]
            point2_y = points[-2][1] 
            point3_y = points[-1][1]
            
            # Từ trên xuống dưới
            if (point1_y < self.line_y - tolerance and 
                point2_y < self.line_y - tolerance and 
                point3_y > self.line_y + tolerance):
                return True
            
            # Từ dưới lên trên  
            if (point1_y > self.line_y + tolerance and 
                point2_y > self.line_y + tolerance and 
                point3_y < self.line_y - tolerance):
                return True
        
        return False
    
    def process_frame(self, frame):
        self.set_line_position(frame.shape[0], frame.shape[1])
        
        # Vẽ line
        if self.line_y is not None and self.line_start and self.line_end:
            cv2.line(frame, self.line_start, self.line_end, (0, 255, 255), 3)
            for x in range(self.line_start[0], self.line_end[0], 80):
                cv2.arrowedLine(frame, (x, self.line_y - 10), 
                               (x, self.line_y + 10), (0, 255, 255), 2)
        
        # Detect
        current_time = time.time()
        if (current_time - self.last_detection) > self.detection_interval:
            results = self.model(frame, conf=0.3, verbose=False, imgsz=416)
            self.last_detection = current_time
            
            current_vehicles = set()
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        center = (center_x, center_y)
                        
                        # Tracking
                        closest = None
                        min_dist = float('inf')
                        
                        for vid, v in self.vehicles.items():
                            if not v.crossed_line:
                                dist = ((center_x - v.center[0])**2 + (center_y - v.center[1])**2)**0.5
                                if dist < min_dist and dist < 150:
                                    min_dist = dist
                                    closest = v
                        
                        if closest:
                            closest.bbox = (x1, y1, x2, y2)
                            closest.center = center
                            current_vehicles.add(closest.id)
                        else:
                            vehicle = type('Vehicle', (), {
                                'id': self.next_id,
                                'bbox': (x1, y1, x2, y2),
                                'center': center,
                                'crossed_line': False
                            })()
                            self.vehicles[self.next_id] = vehicle
                            current_vehicles.add(self.next_id)
                            self.total_detected += 1
                            self.next_id += 1
            
            # Update tracking
            for vid in list(self.vehicles.keys()):
                if vid in current_vehicles:
                    vehicle = self.vehicles[vid]
                    self.track_history[vid].append(vehicle.center)
                    
                    if self.check_crossing(vehicle):
                        if not vehicle.crossed_line:
                            vehicle.crossed_line = True
                            self.total_count += 1
                            self.current_session_count += 1
                            self.total_crossed += 1
                else:
                    if vid in self.vehicles and self.vehicles[vid].crossed_line:
                        del self.vehicles[vid]
        
        # Vẽ vehicles
        for vehicle in self.vehicles.values():
            x1, y1, x2, y2 = vehicle.bbox
            color = (0, 255, 0) if not vehicle.crossed_line else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, str(vehicle.id), (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.circle(frame, vehicle.center, 3, color, -1)
        
        # Vẽ info
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 100), (255, 255, 255), 1)
        
        cv2.putText(frame, f"Qua vach: {self.total_count}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Phat hien: {self.total_detected}", (15, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Phien: {self.current_session_count}", (15, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        status = "THONG THOANG" if self.current_session_count < 10 else "TAC DUONG" if self.current_session_count > 15 else "BINH THUONG"
        color = (0, 255, 0) if self.current_session_count < 10 else (0, 0, 255) if self.current_session_count > 15 else (0, 255, 255)
        cv2.putText(frame, status, (15, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def reset_session(self):
        self.current_session_count = 0
        self.vehicles = {}
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        self.next_id = 0
        self.total_detected = 0
        self.total_crossed = 0
        self.latest_frame = None
    
    def get_status(self):
        return {
            'total_count': self.total_count,
            'current_session_count': self.current_session_count,
            'total_detected': self.total_detected,
            'total_crossed': self.total_crossed,
            'status': "THÔNG THOÁNG" if self.current_session_count < 10 else "TẮC ĐƯỜNG" if self.current_session_count > 15 else "BÌNH THƯỜNG"
        }

# Database setup
def init_db():
    conn = sqlite3.connect('traffic_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  video_name TEXT,
                  start_time TEXT,
                  end_time TEXT,
                  total_vehicles INTEGER,
                  status TEXT)''')
    conn.commit()
    conn.close()

def save_session(video_name, total_vehicles, status):
    conn = sqlite3.connect('traffic_history.db')
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("INSERT INTO sessions (video_name, start_time, end_time, total_vehicles, status) VALUES (?, ?, ?, ?, ?)",
              (video_name, now, now, total_vehicles, status))
    conn.commit()
    conn.close()

def get_sessions():
    conn = sqlite3.connect('traffic_history.db')
    c = conn.cursor()
    c.execute("SELECT * FROM sessions ORDER BY start_time DESC")
    sessions = c.fetchall()
    conn.close()
    return sessions

def process_video_stream(video_path):
    global current_counter, is_processing
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            is_processing = False
            return
        
        print(f"Bắt đầu xử lý video: {video_path}")
        frame_count = 0
        
        # Đảm bảo video có thể đọc được
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while is_processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Video đã hết, quay lại đầu")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Resize frame để giảm kích thước
            frame = cv2.resize(frame, (640, 480))
            
            processed_frame = current_counter.process_frame(frame)
            
            # Encode frame for web display
            success, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            if success:
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                # Store latest frame for web display
                current_counter.latest_frame = frame_base64
                frame_count += 1
                
                if frame_count % 30 == 0:  # Log every 30 frames
                    print(f"Đã xử lý {frame_count} frames, latest_frame size: {len(frame_base64)}")
            else:
                print("Lỗi encode frame!")
            
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
        print("Video processing stopped")
        
    except Exception as e:
        print(f"Lỗi xử lý video: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_processing = False

@app.route('/video_feed')
def video_feed():
    global current_counter
    if current_counter and hasattr(current_counter, 'latest_frame') and current_counter.latest_frame:
        return jsonify({'frame': current_counter.latest_frame})
    return jsonify({'frame': None, 'error': 'No frame available'})

# Routes
@app.route('/')
def index():
    return render_template('simple_index.html')

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global current_counter, is_processing
    
    if is_processing:
        return jsonify({'status': 'error', 'message': 'Đang xử lý video khác'})
    
    # Get video file
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'Không có file video'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Không chọn file'})
    
    # Save video
    uploads_dir = 'uploads'
    os.makedirs(uploads_dir, exist_ok=True)
    video_path = os.path.join(uploads_dir, file.filename)
    file.save(video_path)
    
    # Initialize counter
    current_counter = SimpleTrafficCounter()
    current_counter.current_video = video_path
    current_counter.session_start_time = datetime.now()
    
    # Start processing in thread
    is_processing = True
    processing_thread = threading.Thread(target=process_video_stream, args=(video_path,))
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({'status': 'success', 'message': 'Bắt đầu xử lý video'})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global is_processing, current_counter
    
    if current_counter and current_counter.current_video:
        # Save session
        video_name = os.path.basename(current_counter.current_video)
        status = current_counter.get_status()['status']
        save_session(video_name, current_counter.current_session_count, status)
        
        # Clean up
        if os.path.exists(current_counter.current_video):
            os.remove(current_counter.current_video)
    
    is_processing = False
    current_counter = None
    
    return jsonify({'status': 'success', 'message': 'Đã dừng xử lý'})

@app.route('/get_status')
def get_status():
    global current_counter
    if current_counter:
        return jsonify(current_counter.get_status())
    return jsonify({'status': 'Không hoạt động'})

@app.route('/reset_session', methods=['POST'])
def reset_session():
    global current_counter
    if current_counter:
        current_counter.reset_session()
        return jsonify({'status': 'success', 'message': 'Đã reset phiên'})
    return jsonify({'status': 'error', 'message': 'Không có phiên nào đang hoạt động'})

@app.route('/set_line', methods=['POST'])
def set_line():
    global current_counter
    data = request.json
    start_point = (data['start_x'], data['start_y'])
    end_point = (data['end_x'], data['end_y'])
    
    if current_counter:
        current_counter.set_custom_line(start_point, end_point)
        return jsonify({'status': 'success', 'message': 'Đã cập nhật vạch đếm'})
    return jsonify({'status': 'error', 'message': 'Không có phiên nào đang hoạt động'})

@app.route('/history')
def history():
    sessions = get_sessions()
    return render_template('simple_history.html', sessions=sessions)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5001)
