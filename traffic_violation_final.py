#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script phát hiện vi phạm giao thông sử dụng model COCO gốc (hiệu quả nhất)
Hỗ trợ: Car, Motorbike, Bus, Truck
"""

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# --- CẤU HÌNH VẼ LINE ---
line_pts = []
def mouse_callback(event, x, y, flags, param):
    global line_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(line_pts) < 2:
        line_pts.append((x, y))
        print(f"Point {len(line_pts)}: {(x,y)}")

# --- CẤU HÌNH VIDEO VÀ MODEL ---
VIDEO_SOURCE = r'hi2.mp4'  # Sử dụng video gốc có phương tiện

# Sử dụng model COCO gốc (hiệu quả nhất)
TRAFFIC_MODEL = 'yolov8n.pt'
TRAFFIC_LIGHT_MODEL = 'best_traffic_nano_yolo.pt'

# COCO class mapping cho phương tiện
COCO_CLASSES = {
    2: 'Car',      # Xe ô tô
    3: 'Motorbike', # Xe máy
    5: 'Bus',      # Xe bus
    7: 'Truck',    # Xe tải
}

# Chỉ track các phương tiện
TRACKING_CLASSES = [2, 3, 5, 7]

print("🚀 Traffic Violation Detection System")
print("📱 Supported vehicles: Car, Motorbike, Bus, Truck")

# --- CHỌN VẠCH TRÊN FRAME ĐẦU TIÊN ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Không thể đọc video.")
cv2.namedWindow("Select Detection Line")
cv2.setMouseCallback("Select Detection Line", mouse_callback)

print("\n📍 Hướng dẫn:")
print("   - Click 2 điểm để vẽ vạch phát hiện vi phạm")
print("   - Nhấn 's' để bắt đầu")
print("   - Nhấn 'q' để thoát")

while True:
    disp = first_frame.copy()
    if len(line_pts) >= 1:
        cv2.circle(disp, line_pts[0], 8, (0,255,0), -1)
        cv2.putText(disp, "Point 1", (line_pts[0][0]+10, line_pts[0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if len(line_pts) == 2:
        cv2.line(disp, line_pts[0], line_pts[1], (0,255,0), 3)
        cv2.putText(disp, "Point 2", (line_pts[1][0]+10, line_pts[1][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(disp, "Press 's' to start", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow("Select Detection Line", disp)
    key = cv2.waitKey(1) & 0xFF
    if len(line_pts) == 2 and key == ord('s'):
        break
    if key == ord('q'):
        cap.release(); cv2.destroyAllWindows(); exit("Đã hủy.")
cv2.destroyWindow("Select Detection Line")
cap.release()

# --- TẢI MODEL ---
print("\n🔄 Đang tải models...")
traffic_model = YOLO(TRAFFIC_MODEL)
traffic_light_model = YOLO(TRAFFIC_LIGHT_MODEL)
os.makedirs('vi_pham', exist_ok=True)

print(f"✅ Models loaded successfully")
print(f"🚗 Tracking classes: {[COCO_CLASSES[i] for i in TRACKING_CLASSES]}")

track_history = {}
frame_count = 0
violation_count = 0

def side_of_line(pt, p1, p2):
    """Kiểm tra điểm ở bên nào của đường thẳng"""
    x1,y1 = p1; x2,y2 = p2
    return (x2-x1)*(pt[1]-y1) - (y2-y1)*(pt[0]-x1)

def get_class_name(class_id):
    """Lấy tên class từ class_id"""
    return COCO_CLASSES.get(class_id, f"Unknown_{class_id}")

# --- STREAM với TRACKING & LIGHT DETECTION ---
print("\n🎬 Bắt đầu xử lý video...")
print("💡 Nhấn 'q' trong cửa sổ video để thoát")

for result in traffic_model.track(
        source=VIDEO_SOURCE,
        conf=0.3,  # Confidence thấp hơn để phát hiện nhiều hơn
        iou=0.5,
        classes=TRACKING_CLASSES,
        persist=True,
        stream=True
    ):
    frame_count += 1
    frame = result.orig_img.copy()

    # 1) Vẽ vạch phát hiện
    cv2.line(frame, line_pts[0], line_pts[1], (0,255,0), 3)
    cv2.circle(frame, line_pts[0], 5, (0,255,0), -1)
    cv2.circle(frame, line_pts[1], 5, (0,255,0), -1)

    # 2) Detect đèn giao thông
    tl_res = traffic_light_model(frame, conf=0.3)[0]
    tl_state = None
    for tl_box in tl_res.boxes:
        x1_l,y1_l,x2_l,y2_l = tl_box.xyxy.cpu().numpy().astype(int)[0]
        cls_id = int(tl_box.cls.cpu().item())
        conf_l = float(tl_box.conf.cpu().item())
        name   = traffic_light_model.model.names[cls_id]
        color  = (0,255,0) if name=='green' else (0,0,255) if name=='red' else (255,255,0)
        cv2.rectangle(frame, (x1_l,y1_l),(x2_l,y2_l), color, 2)
        cv2.putText(frame, f"{name}:{conf_l:.2f}", (x1_l,y1_l-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if tl_state is None or conf_l > tl_state[1]:
            tl_state = (name, conf_l)

    light_label = tl_state[0] if tl_state else "no-light"
    light_color = (0,255,0) if light_label=='green' else (0,0,255) if light_label=='red' else (255,255,0)
    cv2.putText(frame, f"Traffic Light: {light_label.upper()}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, light_color, 2)

    # 3) Xử lý từng phương tiện
    for box in result.boxes:
        tid = int(box.id.cpu().item())
        x1,y1,x2,y2 = box.xyxy.cpu().numpy().astype(int)[0]
        cx = (x1 + x2)//2
        cy = y2  # Sử dụng điểm dưới cùng làm reference
        class_id = int(box.cls.cpu().item())
        class_name = get_class_name(class_id)
        conf = float(box.conf.cpu().item())

        # Khởi tạo history nếu track mới
        if tid not in track_history:
            track_history[tid] = {
                'pt': (cx,cy),
                'crossed': False,
                'violation': False,
                'violation_time': None,
                'class_name': class_name,
                'class_id': class_id,
                'first_seen': frame_count
            }
        rec = track_history[tid]

        # Màu box dựa vào flag violation
        box_color = (0,0,255) if rec['violation'] else (255,0,0)

        # Kiểm tra vượt vạch lần đầu
        if not rec['crossed']:
            s_prev = side_of_line(rec['pt'], line_pts[0], line_pts[1])
            s_curr = side_of_line((cx,cy),    line_pts[0], line_pts[1])
            if s_prev * s_curr < 0:  # Đã vượt vạch
                if light_label == 'red':
                    rec['violation'] = True
                    rec['violation_time'] = time.time()
                    violation_count += 1
                    
                    # Lưu ảnh vi phạm
                    crop = result.orig_img[y1:y2, x1:x2]
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    fname = os.path.join('vi_pham', f"{class_name}_{tid}_{timestamp}.jpg")
                    cv2.imwrite(fname, crop)
                    print(f"🚨 [VI PHAM] {class_name.upper()} ID:{tid} - Saved {fname}")
                rec['crossed'] = True

        # Cập nhật centroid
        rec['pt'] = (cx,cy)

        # Vẽ box, ID, class và confidence
        cv2.rectangle(frame, (x1,y1),(x2,y2), box_color, 2)
        label = f"{class_name} ID:{tid} ({conf:.2f})"
        cv2.putText(frame, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.circle(frame, (cx,cy), 4, box_color, -1)

        # Hiển thị "VI PHAM" trong 2 giây sau khi vi phạm
        if rec['violation'] and rec['violation_time'] is not None:
            if time.time() - rec['violation_time'] <= 2.0:
                cv2.putText(frame, "VI PHAM", (x1, y1-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # 4) Hiển thị thống kê
    total_vehicles = len(track_history)
    
    # Thống kê theo loại phương tiện
    vehicle_stats = {}
    violation_stats = {}
    for v in track_history.values():
        vtype = v['class_name']
        vehicle_stats[vtype] = vehicle_stats.get(vtype, 0) + 1
        if v['violation']:
            violation_stats[vtype] = violation_stats.get(vtype, 0) + 1
    
    # Hiển thị thống kê chính
    cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Violations: {violation_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    # Hiển thị thống kê từng loại xe
    y_offset = 120
    for vtype, count in vehicle_stats.items():
        violations = violation_stats.get(vtype, 0)
        color = (0,255,0) if violations == 0 else (0,0,255)
        cv2.putText(frame, f"{vtype}: {count} ({violations} violations)", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25

    # Hiển thị frame info
    cv2.putText(frame, f"Frame: {frame_count}", (frame.shape[1]-150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Traffic Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# --- BÁO CÁO CUỐI ---
print(f"\n📊 BÁO CÁO CUỐI:")
print(f"   📹 Frames processed: {frame_count}")
print(f"   🚗 Total vehicles tracked: {len(track_history)}")
print(f"   🚨 Total violations detected: {violation_count}")
print(f"   📁 Evidence saved in: vi_pham/")

if violation_count > 0:
    print(f"\n🚨 CHI TIẾT VI PHẠM:")
    for tid, rec in track_history.items():
        if rec['violation']:
            print(f"   - {rec['class_name']} ID:{tid} (seen at frame {rec['first_seen']})")

print(f"\n✅ Hoàn thành!")
