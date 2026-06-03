#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script training YOLOv8 cho phát hiện giao thông
Sử dụng dataset TrafficData_YOLO_2.v1i.yolov8 với các class: Bus, Car, Cycle, Truck, Van
"""

import os
import yaml
from ultralytics import YOLO
import torch

def main():
    print("=== TRAINING YOLOV8 TRAFFIC DETECTION MODEL ===")
    
    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Đường dẫn dataset
    dataset_path = "TrafficData_YOLO_2.v1i.yolov8/data.yaml"
    
    # Kiểm tra file dataset
    if not os.path.exists(dataset_path):
        print(f"❌ Không tìm thấy dataset: {dataset_path}")
        return
    
    # Đọc cấu hình dataset
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"📊 Dataset classes: {dataset_config['names']}")
    print(f"📊 Number of classes: {dataset_config['nc']}")
    
    # Tạo model YOLOv8n (nano - nhẹ nhất)
    print("\n🚀 Khởi tạo YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # Sử dụng pre-trained weights
    
    # Cấu hình training (simplified for compatibility)
    training_args = {
        'data': dataset_path,
        'epochs': 30,           # Số epoch
        'imgsz': 640,           # Kích thước ảnh
        'batch': 8,             # Batch size (giảm để tránh lỗi)
        'device': device,       # Device
        'workers': 2,           # Số worker threads
        'project': 'traffic_training',  # Thư mục kết quả
        'name': 'traffic_yolov8n',      # Tên experiment
        'save_period': 5,       # Lưu checkpoint mỗi 5 epochs
        'patience': 10,         # Early stopping patience
        'lr0': 0.01,           # Learning rate ban đầu
        'lrf': 0.01,           # Learning rate cuối
        'momentum': 0.937,      # Momentum
        'weight_decay': 0.0005, # Weight decay
        'warmup_epochs': 2,     # Warmup epochs
        'warmup_momentum': 0.8, # Warmup momentum
        'warmup_bias_lr': 0.1,  # Warmup bias lr
        'box': 7.5,            # Box loss gain
        'cls': 0.5,            # Class loss gain
        'dfl': 1.5,            # DFL loss gain
        'val': True,           # Validate during training
        'plots': True,         # Tạo plots
        'save': True,          # Save checkpoints
        'augment': True,       # Augment images
    }
    
    print("🏋️ Bắt đầu training...")
    print(f"📁 Results sẽ được lưu tại: traffic_training/traffic_yolov8n/")
    
    try:
        # Bắt đầu training
        results = model.train(**training_args)
        
        print("\n✅ Training hoàn thành!")
        print(f"📁 Model được lưu tại: traffic_training/traffic_yolov8n/weights/best.pt")
        print(f"📁 Model cuối: traffic_training/traffic_yolov8n/weights/last.pt")
        
        # Validate model
        print("\n🔍 Đang validate model...")
        val_results = model.val()
        
        print(f"\n📊 Kết quả validation:")
        print(f"   - mAP50: {val_results.box.map50:.3f}")
        print(f"   - mAP50-95: {val_results.box.map:.3f}")
        
        # Test với một vài ảnh
        print("\n🧪 Testing model với ảnh mẫu...")
        test_images = [
            "TrafficData_YOLO_2.v1i.yolov8/test/images",
            "TrafficData_YOLO_2.v1i.yolov8/valid/images"
        ]
        
        for test_dir in test_images:
            if os.path.exists(test_dir):
                test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:3]
                for img_file in test_files:
                    img_path = os.path.join(test_dir, img_file)
                    results = model.predict(img_path, save=True, conf=0.5)
                    print(f"   ✅ Tested: {img_file}")
        
        print("\n🎉 Training và testing hoàn thành!")
        print("💡 Để sử dụng model mới, thay đổi đường dẫn model trong traffic_light_violation.py")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
