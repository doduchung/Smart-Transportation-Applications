#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script test model đã train với các confidence threshold khác nhau
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

def test_model_with_different_confidences():
    print("=== TEST TRAINED MODEL WITH DIFFERENT CONFIDENCES ===")
    
    # Đường dẫn model và video
    model_path = 'traffic_training_small/traffic_yolov8n_small/weights/best.pt'
    video_path = 'hi2.mp4'  # Sử dụng video gốc có phương tiện
    
    # Kiểm tra file
    if not os.path.exists(model_path):
        print(f"❌ Model không tồn tại: {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ Video không tồn tại: {video_path}")
        return
    
    # Load model
    print(f"🔄 Đang load model: {model_path}")
    model = YOLO(model_path)
    
    # Class names
    class_names = {0: 'Bus', 1: 'Car', 2: 'Cycle', 3: 'Truck', 4: 'Van'}
    
    # Test với các confidence khác nhau
    confidences = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for conf in confidences:
        print(f"\n🔍 Testing với confidence = {conf}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        detection_count = 0
        
        # Chỉ test 50 frame đầu
        while frame_count < 50:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Predict
            results = model(frame, conf=conf, verbose=False)
            
            # Đếm detections
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    detection_count += len(result.boxes)
        
        cap.release()
        
        print(f"   📊 Frames: {frame_count}, Detections: {detection_count}")
        if detection_count > 0:
            print(f"   ✅ Model hoạt động tốt với confidence = {conf}")
        else:
            print(f"   ⚠️ Không có detection với confidence = {conf}")

def test_model_with_single_image():
    """Test model với một ảnh từ dataset"""
    print("\n=== TEST MODEL WITH DATASET IMAGES ===")
    
    model_path = 'traffic_training_small/traffic_yolov8n_small/weights/best.pt'
    model = YOLO(model_path)
    
    # Tìm một vài ảnh từ dataset để test
    dataset_images = [
        "TrafficData_YOLO_2.v1i.yolov8/test/images",
        "Yolov8 Traffic.v1i.yolov8/valid/images"
    ]
    
    for dataset_dir in dataset_images:
        if os.path.exists(dataset_dir):
            print(f"\n📁 Testing với ảnh từ: {dataset_dir}")
            images = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')][:3]
            
            for img_file in images:
                img_path = os.path.join(dataset_dir, img_file)
                
                # Test với confidence thấp
                results = model(img_path, conf=0.1, save=True, project='test_results')
                
                # Đếm detections
                detection_count = 0
                for result in results:
                    if result.boxes is not None:
                        detection_count += len(result.boxes)
                
                print(f"   🖼️ {img_file}: {detection_count} detections")

def compare_models():
    """So sánh model trained với model gốc COCO"""
    print("\n=== COMPARE TRAINED MODEL VS COCO MODEL ===")
    
    # Load cả hai model
    trained_model = YOLO('traffic_training_small/traffic_yolov8n_small/weights/best.pt')
    coco_model = YOLO('yolov8n.pt')
    
    # Test video
    video_path = 'hi2.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # Lấy frame đầu tiên
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Không thể đọc video")
        return
    
    print("🔍 Testing frame đầu tiên...")
    
    # Test trained model
    trained_results = trained_model(frame, conf=0.1, verbose=False)
    trained_detections = 0
    for result in trained_results:
        if result.boxes is not None:
            trained_detections += len(result.boxes)
    
    # Test COCO model (chỉ car, bus, truck, motorbike)
    coco_results = coco_model(frame, conf=0.1, classes=[2, 3, 5, 7], verbose=False)
    coco_detections = 0
    for result in coco_results:
        if result.boxes is not None:
            coco_detections += len(result.boxes)
    
    print(f"📊 Trained model: {trained_detections} detections")
    print(f"📊 COCO model: {coco_detections} detections")
    
    if trained_detections > coco_detections:
        print("✅ Trained model phát hiện nhiều hơn")
    elif coco_detections > trained_detections:
        print("⚠️ COCO model phát hiện nhiều hơn")
    else:
        print("🔄 Cả hai model có kết quả tương đương")

if __name__ == "__main__":
    test_model_with_different_confidences()
    test_model_with_single_image()
    compare_models()
    
    print("\n🎉 Test hoàn thành!")
    print("💡 Nếu model không phát hiện được gì, có thể cần:")
    print("   1. Training với dataset lớn hơn (TrafficData_YOLO_2)")
    print("   2. Tăng số epochs")
    print("   3. Điều chỉnh learning rate")
    print("   4. Sử dụng model COCO gốc với confidence thấp hơn")
