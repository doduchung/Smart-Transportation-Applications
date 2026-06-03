#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script training YOLOv8 cho dataset nhỏ hơn
Sử dụng dataset Yolov8 Traffic.v1i.yolov8 với các class: Bike, Bus, Car, Motobike, Truck
"""

import os
import yaml
from ultralytics import YOLO
import torch

def main():
    print("=== TRAINING YOLOV8 TRAFFIC DETECTION MODEL (SMALL DATASET) ===")
    
    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Đường dẫn dataset
    dataset_path = "Yolov8 Traffic.v1i.yolov8/data.yaml"
    
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
    
    # Cấu hình training cho dataset nhỏ (ít epochs hơn)
    training_args = {
        'data': dataset_path,
        'epochs': 30,           # Ít epochs hơn cho dataset nhỏ
        'imgsz': 640,           # Kích thước ảnh
        'batch': 8,             # Batch size nhỏ hơn
        'device': device,       # Device
        'workers': 2,           # Ít workers hơn
        'project': 'traffic_training_small',  # Thư mục kết quả
        'name': 'traffic_yolov8n_small',      # Tên experiment
        'save_period': 5,       # Lưu checkpoint mỗi 5 epochs
        'patience': 10,         # Early stopping patience
        'lr0': 0.01,           # Learning rate ban đầu
        'lrf': 0.01,           # Learning rate cuối
        'momentum': 0.937,      # Momentum
        'weight_decay': 0.0005, # Weight decay
        'warmup_epochs': 2,     # Ít warmup epochs hơn
        'warmup_momentum': 0.8, # Warmup momentum
        'warmup_bias_lr': 0.1,  # Warmup bias lr
        'box': 7.5,            # Box loss gain
        'cls': 0.5,            # Class loss gain
        'dfl': 1.5,            # DFL loss gain
        'val': True,           # Validate during training
        'plots': True,         # Tạo plots
        'save': True,          # Save checkpoints
        'augment': True,       # Augment images (quan trọng cho dataset nhỏ)
        'copy_paste': 0.0,     # Copy-paste augmentation
        'degrees': 10.0,       # Rotation degrees
        'translate': 0.1,      # Translation
        'scale': 0.5,          # Scaling
        'shear': 2.0,          # Shearing
        'perspective': 0.0,    # Perspective
        'flipud': 0.0,         # Flip up-down
        'fliplr': 0.5,         # Flip left-right
        'mosaic': 1.0,         # Mosaic augmentation
        'mixup': 0.0,          # Mixup augmentation
        'hsv_h': 0.015,        # HSV-H augmentation
        'hsv_s': 0.7,          # HSV-S augmentation
        'hsv_v': 0.4,          # HSV-V augmentation
    }
    
    print("🏋️ Bắt đầu training...")
    print(f"📁 Results sẽ được lưu tại: traffic_training_small/traffic_yolov8n_small/")
    
    try:
        # Bắt đầu training
        results = model.train(**training_args)
        
        print("\n✅ Training hoàn thành!")
        print(f"📁 Model được lưu tại: traffic_training_small/traffic_yolov8n_small/weights/best.pt")
        print(f"📁 Model cuối: traffic_training_small/traffic_yolov8n_small/weights/last.pt")
        
        # Validate model
        print("\n🔍 Đang validate model...")
        val_results = model.val()
        
        print(f"\n📊 Kết quả validation:")
        print(f"   - mAP50: {val_results.box.map50:.3f}")
        print(f"   - mAP50-95: {val_results.box.map:.3f}")
        
        print("\n🎉 Training hoàn thành!")
        print("💡 Model này phù hợp cho việc test nhanh với dataset nhỏ")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
