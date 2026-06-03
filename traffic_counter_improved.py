#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hệ thống đếm phương tiện cải tiến
- Thu gọn text, cải thiện nhận diện
- Cho phép kẻ line tùy chỉnh
- Tăng tốc video mượt mà
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO
import os
import json
from datetime import datetime
from collections import defaultdict, deque

class ImprovedTrafficCounter:
    def __init__(self, model_path='traffic_training_small/traffic_yolov8n_small/weights/best.pt'):
        self.model = YOLO(model_path)
        
        # Variables
        self.vehicles = {}
        self.next_id = 0
        self.track_history = defaultdict(lambda: deque(maxlen=8))
        
        # Line có thể tùy chỉnh
        self.line_y = None
        self.line_start = None
        self.line_end = None
        self.drawing_line = False
        
        # Counts
        self.total_count = 0
        self.current_session_count = 0
        
        # Performance - Tối ưu cho mượt mà
        self.frame_count = 0
        self.last_detection = 0
        self.detection_interval = 0.15  # Detect thường xuyên hơn để track tốt
        
        # Debug cho line crossing
        self.debug_mode = False
        
        # Thống kê để debug
        self.total_detected = 0
        self.total_crossed = 0
        
        # Lịch sử
        self.history = []
        
    def set_line_position(self, frame_height, frame_width):
        """Đặt vạch kẻ mặc định"""
        if self.line_y is None:
            self.line_y = frame_height // 2
            self.line_start = (50, self.line_y)
            self.line_end = (frame_width - 50, self.line_y)
    
    def draw_line(self, frame):
        """Vẽ vạch kẻ đơn giản"""
        if self.line_y is not None and self.line_start and self.line_end:
            # Vẽ line chính
            cv2.line(frame, self.line_start, self.line_end, (0, 255, 255), 3)
            
            # Vẽ tolerance zone (để debug)
            if self.debug_mode:
                cv2.line(frame, (self.line_start[0], self.line_y - 5), 
                        (self.line_end[0], self.line_y - 5), (255, 255, 255), 1)
                cv2.line(frame, (self.line_start[0], self.line_y + 5), 
                        (self.line_end[0], self.line_y + 5), (255, 255, 255), 1)
            
            # Vẽ mũi tên đơn giản
            for x in range(self.line_start[0], self.line_end[0], 80):
                cv2.arrowedLine(frame, (x, self.line_y - 10), 
                               (x, self.line_y + 10), (0, 255, 255), 2)
    
    def draw_vehicle(self, frame, vehicle):
        """Vẽ phương tiện đơn giản"""
        x1, y1, x2, y2 = vehicle.bbox
        color = (0, 255, 0)  # Xanh lá
        
        # Bounding box mỏng
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        # Label nhỏ
        cv2.putText(frame, str(vehicle.id), (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Center point nhỏ
        cv2.circle(frame, vehicle.center, 2, color, -1)
    
    def check_crossing(self, vehicle):
        """Kiểm tra đi qua vạch kẻ - logic cải thiện"""
        if self.line_y is None or vehicle.id not in self.track_history:
            return False
        
        points = list(self.track_history[vehicle.id])
        if len(points) < 3:  # Cần ít nhất 3 điểm để kiểm tra chính xác
            return False
        
        # Kiểm tra nhiều điểm để tránh false positive
        tolerance = 10  # Tăng tolerance
        
        # Lấy 3 điểm cuối cùng để kiểm tra xu hướng
        if len(points) >= 3:
            point1_y = points[-3][1]
            point2_y = points[-2][1] 
            point3_y = points[-1][1]
            
            # Kiểm tra xu hướng đi qua line
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
        
        # Kiểm tra đơn giản với 2 điểm cuối
        if len(points) >= 2:
            prev_y = points[-2][1]
            curr_y = points[-1][1]
            
            # Đi qua line với khoảng cách đủ lớn
            if abs(prev_y - curr_y) > 20:  # Phải di chuyển đủ xa
                if prev_y < (self.line_y - tolerance) and curr_y > (self.line_y + tolerance):
                    return True
                if prev_y > (self.line_y + tolerance) and curr_y < (self.line_y - tolerance):
                    return True
        
        return False
    
    def update_count(self, vehicle):
        """Cập nhật đếm"""
        if not vehicle.crossed_line:
            vehicle.crossed_line = True
            self.total_count += 1
            self.current_session_count += 1
            self.total_crossed += 1
            print(f"🚗 Xe {vehicle.id} đi qua vạch kẻ - Tổng: {self.total_count}")
            print(f"📊 Thống kê: Phát hiện {self.total_detected} xe, đã qua vạch {self.total_crossed} xe")
    
    def get_traffic_status(self):
        """Lấy trạng thái giao thông"""
        if self.current_session_count < 10:
            return "THÔNG THOÁNG", (0, 255, 0)
        elif self.current_session_count <= 15:
            return "BÌNH THƯỜNG", (0, 255, 255)
        else:
            return "TẮC ĐƯỜNG!", (0, 0, 255)
    
    def draw_info_compact(self, frame, status_text, status_color):
        """Vẽ thông tin gọn gàng"""
        # Background nhỏ hơn
        cv2.rectangle(frame, (10, 10), (280, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (280, 100), (255, 255, 255), 1)
        
        # Text nhỏ
        cv2.putText(frame, f"Qua vach: {self.total_count}", (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Phat hien: {self.total_detected}", (15, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(frame, f"Phiên: {self.current_session_count}", (15, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, status_text, (15, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    def process_frame(self, frame):
        """Xử lý frame tối ưu"""
        self.frame_count += 1
        
        # Đặt line
        self.set_line_position(frame.shape[0], frame.shape[1])
        self.draw_line(frame)
        
        # Detect với tần suất vừa phải để mượt mà
        current_time = time.time()
        if (current_time - self.last_detection) > self.detection_interval:
            # Tối ưu detection cho mượt mà
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
                        
                        # Tracking với threshold lớn hơn
                        closest = None
                        min_dist = float('inf')
                        
                        for vid, v in self.vehicles.items():
                            if not v.crossed_line:
                                dist = ((center_x - v.center[0])**2 + (center_y - v.center[1])**2)**0.5
                                if dist < min_dist and dist < 150:  # Tăng threshold để track tốt hơn
                                    min_dist = dist
                                    closest = v
                        
                        if closest:
                            closest.bbox = (x1, y1, x2, y2)
                            closest.center = center
                            current_vehicles.add(closest.id)
                        else:
                            # Tạo vehicle mới
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
                            print(f"🆕 Phát hiện xe mới ID: {self.next_id - 1}")
            
            # Update tracking
            for vid in list(self.vehicles.keys()):
                if vid in current_vehicles:
                    vehicle = self.vehicles[vid]
                    self.track_history[vid].append(vehicle.center)
                    
                    if self.check_crossing(vehicle):
                        self.update_count(vehicle)
                else:
                    if vid in self.vehicles and self.vehicles[vid].crossed_line:
                        del self.vehicles[vid]
        
        # Vẽ vehicles
        for vehicle in self.vehicles.values():
            self.draw_vehicle(frame, vehicle)
        
        # Vẽ info compact
        status_text, status_color = self.get_traffic_status()
        self.draw_info_compact(frame, status_text, status_color)
        
        return frame
    
    def reset_session(self):
        """Reset phiên hiện tại"""
        # Lưu phiên trước khi reset (nếu có dữ liệu)
        if self.current_session_count > 0 and hasattr(self, 'current_video_name'):
            self.save_session(self.current_video_name)
        
        self.current_session_count = 0
        self.vehicles = {}
        self.track_history = defaultdict(lambda: deque(maxlen=8))
        self.next_id = 0
        self.total_detected = 0
        self.total_crossed = 0
        self.session_start_time = time.time()
    
    def save_session(self, video_name):
        """Lưu kết quả phiên với thông tin chi tiết"""
        status_text, status_color = self.get_traffic_status()
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'video_name': video_name,
            'vehicles_passed_line': self.current_session_count,  # Xe đã qua vạch
            'vehicles_detected': self.total_detected,  # Xe đã đếm/phát hiện
            'traffic_status': status_text,  # Tắc đường hay thông thoáng
            'total_count_all_time': self.total_count,  # Tổng xe qua vạch từ trước
            'session_duration': self.get_session_duration()  # Thời gian xử lý
        }
        self.history.append(session_data)
        
        # Lưu vào file
        with open('traffic_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Đã lưu lịch sử: {self.current_session_count} xe qua vạch, {self.total_detected} xe phát hiện, {status_text}")
    
    def get_session_duration(self):
        """Tính thời gian xử lý phiên"""
        if hasattr(self, 'session_start_time'):
            duration = time.time() - self.session_start_time
            return f"{int(duration//60)}m {int(duration%60)}s"
        return "N/A"

class ImprovedTrafficGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hệ Thống Đếm Phương Tiện - Cải Tiến")
        self.root.geometry("900x700")
        
        self.counter = ImprovedTrafficCounter()
        self.cap = None
        self.is_running = False
        self.current_video = None
        
        # Line drawing
        self.drawing_mode = False
        self.line_points = []
        
        self.setup_ui()
        self.load_history()
    
    def setup_ui(self):
        """Thiết lập giao diện"""
        # Title
        title_label = tk.Label(self.root, text="🚦 HỆ THỐNG ĐẾM PHƯƠNG TIỆN CẢI TIẾN", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Video selection frame
        video_frame = ttk.LabelFrame(self.root, text="Chọn Video", padding=10)
        video_frame.pack(fill="x", padx=10, pady=5)
        
        self.video_label = tk.Label(video_frame, text="Chưa chọn video")
        self.video_label.pack(side="left", padx=5)
        
        ttk.Button(video_frame, text="Chọn Video", 
                  command=self.select_video).pack(side="right", padx=5)
        
        # Control frame
        control_frame = ttk.LabelFrame(self.root, text="Điều Khiển", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        self.play_button = ttk.Button(control_frame, text="▶ Bắt Đầu", 
                                     command=self.toggle_play, state="disabled")
        self.play_button.pack(side="left", padx=5)
        
        ttk.Button(control_frame, text="🔄 Reset", 
                  command=self.reset_session).pack(side="left", padx=5)
        
        ttk.Button(control_frame, text="📏 Kẻ Line", 
                  command=self.toggle_drawing).pack(side="left", padx=5)
        
        self.debug_button = ttk.Button(control_frame, text="🔍 Debug", 
                                      command=self.toggle_debug)
        self.debug_button.pack(side="left", padx=5)
        
        ttk.Button(control_frame, text="💾 Lưu Kết Quả", 
                  command=self.save_session).pack(side="left", padx=5)
        
        ttk.Button(control_frame, text="📊 Thống Kê", 
                  command=self.show_statistics).pack(side="left", padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Trạng Thái Chi Tiết", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        # Tạo grid layout cho status
        status_left = tk.Frame(status_frame)
        status_left.pack(side="left", fill="both", expand=True)
        
        status_right = tk.Frame(status_frame)
        status_right.pack(side="right", fill="both", expand=True)
        
        self.total_label = tk.Label(status_left, text="Tổng xe qua vạch: 0", 
                                   font=("Arial", 11, "bold"))
        self.total_label.pack(anchor="w", pady=2)
        
        self.session_label = tk.Label(status_left, text="Phiên hiện tại: 0", 
                                     font=("Arial", 11))
        self.session_label.pack(anchor="w", pady=2)
        
        self.detected_label = tk.Label(status_left, text="Xe phát hiện: 0", 
                                      font=("Arial", 11))
        self.detected_label.pack(anchor="w", pady=2)
        
        self.status_label = tk.Label(status_right, text="🚦 THÔNG THOÁNG", 
                                    font=("Arial", 12, "bold"), fg="green")
        self.status_label.pack(anchor="w", pady=2)
        
        self.duration_label = tk.Label(status_right, text="Thời gian: 0m 0s", 
                                      font=("Arial", 10))
        self.duration_label.pack(anchor="w", pady=2)
        
        self.accuracy_label = tk.Label(status_right, text="Độ chính xác: N/A", 
                                      font=("Arial", 10))
        self.accuracy_label.pack(anchor="w", pady=2)
        
        # History frame
        history_frame = ttk.LabelFrame(self.root, text="Lịch Sử Chi Tiết", padding=10)
        history_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Treeview for history với nhiều cột hơn
        columns = ("Thời gian", "Video", "Xe qua vạch", "Xe phát hiện", "Trạng thái", "Thời gian xử lý")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=6)
        
        # Thiết lập độ rộng cột
        column_widths = {"Thời gian": 120, "Video": 100, "Xe qua vạch": 80, "Xe phát hiện": 80, "Trạng thái": 100, "Thời gian xử lý": 90}
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=column_widths[col], anchor="center")
        
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        self.history_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Instructions
        self.instructions_label = tk.Label(self.root, 
                              text="📋 Hướng dẫn: Nhấn 'Kẻ Line' rồi click 2 điểm trên video để vẽ vạch đếm",
                              font=("Arial", 10))
        self.instructions_label.pack(pady=5)
    
    def select_video(self):
        """Chọn video"""
        file_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=file_types
        )
        
        if filename:
            self.current_video = filename
            self.video_label.config(text=f"Video: {os.path.basename(filename)}")
            self.play_button.config(state="normal")
    
    def toggle_play(self):
        """Bắt đầu/dừng video"""
        if not self.is_running:
            self.start_processing()
        else:
            self.stop_processing()
    
    def toggle_drawing(self):
        """Bật/tắt chế độ vẽ line"""
        self.drawing_mode = not self.drawing_mode
        if self.drawing_mode:
            self.instructions_label.config(text="🔴 CHẾ ĐỘ KẺ LINE: Nhấn chuột trái 2 lần trên video!")
            self.line_points = []  # Reset points
        else:
            self.instructions_label.config(text="📋 Hướng dẫn: Nhấn 'Kẻ Line' rồi click 2 điểm trên video để vẽ vạch đếm")
    
    def toggle_debug(self):
        """Bật/tắt chế độ debug"""
        self.counter.debug_mode = not self.counter.debug_mode
        if self.counter.debug_mode:
            self.debug_button.config(text="🔍 Debug ON")
            messagebox.showinfo("Debug", "Chế độ debug đã bật - sẽ hiển thị tolerance zone")
        else:
            self.debug_button.config(text="🔍 Debug")
            messagebox.showinfo("Debug", "Chế độ debug đã tắt")
    
    def start_processing(self):
        """Bắt đầu xử lý video"""
        if not self.current_video:
            messagebox.showerror("Lỗi", "Vui lòng chọn video trước!")
            return
        
        self.cap = cv2.VideoCapture(self.current_video)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở video!")
            return
        
        # Tối ưu video cho mượt mà
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # FPS cao
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Giảm độ phân giải để mượt hơn
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Lưu tên video cho session
        self.counter.current_video_name = os.path.basename(self.current_video)
        
        # Bắt đầu session mới
        self.counter.session_start_time = time.time()
        
        self.is_running = True
        self.play_button.config(text="⏸ Dừng")
        
        # Chạy trong thread riêng
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        """Dừng xử lý video và tự động lưu lịch sử"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        # Tự động lưu lịch sử khi dừng
        if hasattr(self.counter, 'current_video_name') and self.counter.current_session_count > 0:
            self.counter.save_session(self.counter.current_video_name)
            self.load_history()  # Cập nhật lại giao diện
            messagebox.showinfo("Thông báo", 
                f"✅ Đã tự động lưu lịch sử!\n"
                f"📊 Xe qua vạch: {self.counter.current_session_count}\n"
                f"🚗 Xe phát hiện: {self.counter.total_detected}\n"
                f"🚦 Trạng thái: {self.counter.get_traffic_status()[0]}")
        
        self.play_button.config(text="▶ Bắt Đầu")
        cv2.destroyAllWindows()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Xử lý sự kiện chuột để vẽ line"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing_mode:
                if len(self.line_points) == 0:
                    self.line_points = [(x, y)]
                    print(f"✅ Điểm 1: ({x}, {y}) - Nhấn điểm thứ 2")
                elif len(self.line_points) == 1:
                    self.line_points.append((x, y))
                    print(f"✅ Điểm 2: ({x}, {y})")
                    
                    # Cập nhật line
                    self.counter.line_start = self.line_points[0]
                    self.counter.line_end = self.line_points[1]
                    self.counter.line_y = (self.line_points[0][1] + self.line_points[1][1]) // 2
                    
                    print(f"✅ Đã vẽ vạch đếm từ ({self.line_points[0]}) đến ({self.line_points[1]})!")
                    print(f"✅ Line Y = {self.counter.line_y}")
                    self.line_points = []
                    self.drawing_mode = False
                    
                    # Cập nhật UI
                    self.root.after(0, lambda: self.instructions_label.config(
                        text="✅ Đã vẽ vạch đếm thành công! Có thể bắt đầu đếm xe."))
    
    def process_video(self):
        """Xử lý video trong thread riêng"""
        cv2.namedWindow('Traffic Counter - Improved', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Traffic Counter - Improved', self.mouse_callback)
        
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                # Hết video, quay lại đầu
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Xử lý frame
            processed = self.counter.process_frame(frame)
            
            # Hiển thị
            cv2.imshow('Traffic Counter - Improved', processed)
            
            # Cập nhật UI
            self.root.after(0, self.update_ui)
            
            key = cv2.waitKey(30) & 0xFF  # Tăng delay để mượt mà hơn
            if key == ord('q'):
                break
        
        self.root.after(0, self.stop_processing)
    
    def update_ui(self):
        """Cập nhật giao diện với thông tin chi tiết"""
        self.total_label.config(text=f"Tổng xe qua vạch: {self.counter.total_count}")
        self.session_label.config(text=f"Phiên hiện tại: {self.counter.current_session_count}")
        self.detected_label.config(text=f"Xe phát hiện: {self.counter.total_detected}")
        
        status_text, status_color = self.counter.get_traffic_status()
        
        # Đổi màu theo trạng thái
        if status_text == "THÔNG THOÁNG":
            color = "green"
            emoji = "🚦"
        elif status_text == "BÌNH THƯỜNG":
            color = "orange"
            emoji = "⚠️"
        else:
            color = "red"
            emoji = "🚨"
        
        self.status_label.config(text=f"{emoji} {status_text}", fg=color)
        
        # Cập nhật thời gian
        if hasattr(self.counter, 'session_start_time'):
            duration = time.time() - self.counter.session_start_time
            self.duration_label.config(text=f"Thời gian: {int(duration//60)}m {int(duration%60)}s")
        
        # Tính độ chính xác
        if self.counter.total_detected > 0:
            accuracy = (self.counter.current_session_count / self.counter.total_detected) * 100
            self.accuracy_label.config(text=f"Độ chính xác: {accuracy:.1f}%")
        else:
            self.accuracy_label.config(text="Độ chính xác: N/A")
    
    def reset_session(self):
        """Reset phiên hiện tại"""
        self.counter.reset_session()
        self.update_ui()
        messagebox.showinfo("Thông báo", "Đã reset phiên hiện tại!")
    
    def save_session(self):
        """Lưu kết quả phiên"""
        if not self.current_video:
            messagebox.showerror("Lỗi", "Chưa có video nào được chọn!")
            return
        
        video_name = os.path.basename(self.current_video)
        self.counter.save_session(video_name)
        self.load_history()
        messagebox.showinfo("Thông báo", "Đã lưu kết quả!")
    
    def load_history(self):
        """Tải lịch sử với thông tin chi tiết"""
        # Xóa dữ liệu cũ
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Tải dữ liệu mới
        for session in self.counter.history:
            timestamp = datetime.fromisoformat(session['timestamp']).strftime("%d/%m %H:%M")
            
            # Xử lý dữ liệu cũ (nếu có)
            vehicles_passed = session.get('vehicles_passed_line', session.get('count', 0))
            vehicles_detected = session.get('vehicles_detected', 0)
            traffic_status = session.get('traffic_status', session.get('status', 'N/A'))
            duration = session.get('session_duration', 'N/A')
            
            self.history_tree.insert("", "end", values=(
                timestamp,
                session['video_name'],
                vehicles_passed,
                vehicles_detected,
                traffic_status,
                duration
            ))
    
    def show_statistics(self):
        """Hiển thị thống kê tổng quan"""
        if not self.counter.history:
            messagebox.showinfo("Thống kê", "Chưa có dữ liệu lịch sử!")
            return
        
        # Tính toán thống kê
        total_sessions = len(self.counter.history)
        total_vehicles_passed = sum(session.get('vehicles_passed_line', session.get('count', 0)) for session in self.counter.history)
        total_vehicles_detected = sum(session.get('vehicles_detected', 0) for session in self.counter.history)
        
        # Đếm trạng thái giao thông
        traffic_counts = {'THÔNG THOÁNG': 0, 'BÌNH THƯỜNG': 0, 'TẮC ĐƯỜNG!': 0}
        for session in self.counter.history:
            status = session.get('traffic_status', session.get('status', 'THÔNG THOÁNG'))
            if status in traffic_counts:
                traffic_counts[status] += 1
        
        # Tính trung bình
        avg_vehicles_per_session = total_vehicles_passed / total_sessions if total_sessions > 0 else 0
        accuracy = (total_vehicles_passed / total_vehicles_detected * 100) if total_vehicles_detected > 0 else 0
        
        # Tạo message
        stats_message = f"""
📊 THỐNG KÊ TỔNG QUAN

🔢 Tổng số phiên: {total_sessions}
🚗 Tổng xe qua vạch: {total_vehicles_passed}
👁️ Tổng xe phát hiện: {total_vehicles_detected}
📈 Trung bình/phíên: {avg_vehicles_per_session:.1f} xe
🎯 Độ chính xác: {accuracy:.1f}%

🚦 PHÂN BỐ TRẠNG THÁI:
🟢 Thông thoáng: {traffic_counts['THÔNG THOÁNG']} phiên
🟡 Bình thường: {traffic_counts['BÌNH THƯỜNG']} phiên  
🔴 Tắc đường: {traffic_counts['TẮC ĐƯỜNG!']} phiên

📅 Thời gian: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        """
        
        messagebox.showinfo("Thống Kê Chi Tiết", stats_message)
    
    def run(self):
        """Chạy ứng dụng"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Đóng ứng dụng"""
        self.stop_processing()
        self.root.destroy()

def main():
    app = ImprovedTrafficGUI()
    app.run()

if __name__ == "__main__":
    main()
