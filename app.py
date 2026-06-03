#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Application cho Hệ Thống Đếm Phương Tiện Giao Thông
Sử dụng Flask + OpenCV + YOLO
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import sqlite3
from datetime import datetime
import base64
import threading
import time
from collections import defaultdict, deque
import plotly.graph_objs as go
import plotly.utils
import pandas as pd

app = Flask(__name__)
app.secret_key = 'traffic_counter_secret_key_2024'

# Global variables
current_counter = None
processing_thread = None
is_processing = False

class WebTrafficCounter:
    def __init__(self, model_path='traffic_training_small/traffic_yolov8n_small/weights/best.pt'):
        self.model = YOLO(model_path)
        
        # Variables
        self.vehicles = {}
        self.next_id = 0
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        
        # Line có thể tùy chỉnh
        self.line_y = None
        self.line_start = None
        self.line_end = None
        
        # Counts
        self.total_count = 0
        self.current_session_count = 0
        self.total_detected = 0
        self.total_crossed = 0
        
        # Performance
        self.frame_count = 0
        self.last_detection = 0
        self.detection_interval = 0.2
        
        # Current session info
        self.current_video = None
        self.session_start_time = None
        
    def set_line_position(self, frame_height, frame_width):
        """Đặt vạch kẻ mặc định"""
        if self.line_y is None:
            self.line_y = frame_height // 2
            self.line_start = (50, self.line_y)
            self.line_end = (frame_width - 50, self.line_y)
    
    def set_custom_line(self, start_point, end_point):
        """Đặt vạch kẻ tùy chỉnh"""
        self.line_start = start_point
        self.line_end = end_point
        self.line_y = (start_point[1] + end_point[1]) // 2
    
    def check_crossing(self, vehicle):
        """Kiểm tra đi qua vạch kẻ"""
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
        """Xử lý frame"""
        self.frame_count += 1
        
        # Đặt line
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
        """Reset phiên hiện tại"""
        self.current_session_count = 0
        self.vehicles = {}
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        self.next_id = 0
        self.total_detected = 0
        self.total_crossed = 0
    
    def get_status(self):
        """Lấy trạng thái hiện tại"""
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
                  status TEXT,
                  notes TEXT)''')
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

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global current_counter, processing_thread, is_processing
    
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
    current_counter = WebTrafficCounter()
    current_counter.current_video = video_path
    current_counter.session_start_time = datetime.now()
    
    # Start processing in thread
    is_processing = True
    processing_thread = threading.Thread(target=process_video, args=(video_path,))
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({'status': 'success', 'message': 'Bắt đầu xử lý video'})

def process_video(video_path):
    global current_counter, is_processing
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            is_processing = False
            return
        
        while is_processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            processed_frame = current_counter.process_frame(frame)
            
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
        
    except Exception as e:
        print(f"Lỗi xử lý video: {e}")
    finally:
        is_processing = False

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
    return render_template('history.html', sessions=sessions)

@app.route('/analytics')
def analytics():
    sessions = get_sessions()
    
    if not sessions:
        return render_template('analytics.html', chart_data=None)
    
    # Prepare data for charts
    df = pd.DataFrame(sessions, columns=['id', 'video_name', 'start_time', 'end_time', 'total_vehicles', 'status'])
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['date'] = df['start_time'].dt.date
    
    # Daily statistics
    daily_stats = df.groupby('date').agg({
        'total_vehicles': ['sum', 'mean', 'count'],
        'status': lambda x: (x == 'TẮC ĐƯỜNG').sum()
    }).round(2)
    
    daily_stats.columns = ['Total_Vehicles', 'Avg_Vehicles', 'Sessions', 'Traffic_Jams']
    daily_stats = daily_stats.reset_index()
    
    # Create charts
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['Total_Vehicles'], 
                             mode='lines+markers', name='Tổng xe', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['Avg_Vehicles'], 
                             mode='lines+markers', name='Trung bình xe', line=dict(color='green')))
    fig1.update_layout(title='Thống Kê Xe Theo Ngày', xaxis_title='Ngày', yaxis_title='Số xe')
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=daily_stats['date'], y=daily_stats['Traffic_Jams'], 
                         name='Số lần tắc đường', marker_color='red'))
    fig2.update_layout(title='Tần Suất Tắc Đường Theo Ngày', xaxis_title='Ngày', yaxis_title='Số lần')
    
    fig3 = go.Figure()
    status_counts = df['status'].value_counts()
    fig3.add_trace(go.Pie(labels=status_counts.index, values=status_counts.values, 
                         title="Phân Bố Trạng Thái Giao Thông"))
    
    # Convert to JSON
    chart1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    chart2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    chart3_json = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('analytics.html', 
                         chart1=chart1_json, 
                         chart2=chart2_json, 
                         chart3=chart3_json,
                         daily_stats=daily_stats.to_dict('records'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
