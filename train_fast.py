#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script training YOLOv8 nhanh với cấu hình tối ưu
"""

import os
import yaml
from ultralytics import YOLO
import torch

def main():
    print("=== FAST TRAINING YOLOV8 TRAFFIC MODEL ===")
    
    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Đường dẫn dataset
    dataset_path = "TrafficData_YOLO_2.v1i.yolov8/data.yaml"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Không tìm thấy dataset: {dataset_path}")
        return
    
    # Load model
    print("🚀 Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Cấu hình training NHANH
    training_args = {
        'data': dataset_path,
        'epochs': 10,           # Ít epochs hơn
        'imgsz': 416,           # Kích thước ảnh nhỏ hơn
        'batch': 16,            # Batch size lớn hơn
        'device': device,
        'workers': 4,           # Nhiều workers hơn
        'project': 'fast_training',
        'name': 'traffic_fast',
        'save_period': 2,       # Lưu checkpoint thường xuyên hơn
        'patience': 5,          # Early stopping sớm
        'lr0': 0.02,           # Learning rate cao hơn
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 1,     # Ít warmup
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'val': True,
        'plots': False,         # Tắt plots để nhanh hơn
        'save': True,
        'augment': True,
        'cache': False,         # Không cache để tiết kiệm RAM
        'close_mosaic': 5,      # Đóng mosaic sớm
        'amp': True,           # Mixed precision
    }
    
    print("🏋️ Bắt đầu training NHANH...")
    print(f"📁 Results: fast_training/traffic_fast/")
    
    try:
        # Training
        results = model.train(**training_args)
        
        print("\n✅ Training hoàn thành!")
        print(f"📁 Model: fast_training/traffic_fast/weights/best.pt")
        
        # Validate
        print("🔍 Validating...")
        val_results = model.val()
        
        print(f"\n📊 Kết quả:")
        print(f"   - mAP50: {val_results.box.map50:.3f}")
        print(f"   - mAP50-95: {val_results.box.map:.3f}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
