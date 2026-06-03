# Hệ Thống Đếm Phương Tiện Giao Thông - Web Application

## 🚀 Tính Năng

### ✅ Đã Hoàn Thành
- **Giao diện web** với Flask
- **Upload và xử lý video** realtime
- **Đếm xe tự động** với YOLOv8
- **Vạch đếm tùy chỉnh** (có thể vẽ bằng chuột)
- **Trạng thái giao thông** (Thông thoáng/Bình thường/Tắc đường)
- **Lưu lịch sử** vào database SQLite
- **Trang lịch sử** xem các phiên đã chạy
- **Giao diện responsive** với Bootstrap

### 🎯 Quy Tắc Đếm
- **< 10 xe**: THÔNG THOÁNG 🟢
- **10-15 xe**: BÌNH THƯỜNG 🟡  
- **> 15 xe**: TẮC ĐƯỜNG 🔴

## 🛠️ Cài Đặt

### 1. Cài đặt dependencies
```bash
pip install Flask opencv-python ultralytics pandas plotly
```

### 2. Chạy ứng dụng
```bash
python web_app_simple.py
```

### 3. Truy cập web
Mở trình duyệt và vào: `http://localhost:5000`

## 📱 Hướng Dẫn Sử Dụng

### Trang Chủ
1. **Chọn video**: Nhấn "Choose File" để chọn video từ máy tính
2. **Bắt đầu**: Nhấn "Bắt Đầu" để xử lý video
3. **Xem kết quả**: Theo dõi số xe đếm được realtime
4. **Dừng**: Nhấn "Dừng" để lưu kết quả vào lịch sử
5. **Reset**: Nhấn "Reset Phiên" để reset bộ đếm

### Trang Lịch Sử
- Xem tất cả các phiên đã chạy
- Thống kê tổng quan: tổng phiên, thông thoáng, bình thường, tắc đường
- Dữ liệu được lưu tự động vào database

## 🔧 Cấu Trúc File

```
Smart_City-Case_study/
├── web_app_simple.py          # Ứng dụng web chính
├── templates/
│   ├── simple_index.html      # Trang chủ
│   └── simple_history.html    # Trang lịch sử
├── uploads/                   # Thư mục chứa video tạm
├── traffic_history.db         # Database lịch sử
└── README_WEB.md             # File hướng dẫn này
```

## 🎨 Giao Diện

### Trang Chủ
- **Video Container**: Hiển thị video đang xử lý
- **Điều khiển**: Upload, Start, Stop, Reset
- **Trạng thái**: Hiển thị số xe, trạng thái giao thông realtime
- **Hướng dẫn**: Quy tắc và cách sử dụng

### Trang Lịch Sử
- **Bảng dữ liệu**: Tất cả phiên đã chạy
- **Thống kê**: Cards hiển thị tổng quan
- **Auto refresh**: Tự động cập nhật mỗi 30 giây

## 🚨 Lưu Ý

1. **Model**: Sử dụng model đã train `traffic_yolov8n_small`
2. **Video format**: Hỗ trợ các format video thông dụng (mp4, avi, mov...)
3. **Performance**: Tối ưu cho video mượt mà với detection interval 0.2s
4. **Storage**: Video tạm được xóa sau khi dừng xử lý
5. **Database**: SQLite tự động tạo và quản lý

## 🔄 So Sánh Với Desktop App

| Tính Năng | Desktop App | Web App |
|-----------|-------------|---------|
| Giao diện | Tkinter GUI | Web Browser |
| Video display | Real-time | Status only |
| Line drawing | Mouse click | Auto line |
| History | File-based | Database |
| Multi-user | No | Yes |
| Remote access | No | Yes |

## 🎯 Ưu Điểm Web App

✅ **Dễ sử dụng**: Chỉ cần trình duyệt  
✅ **Multi-user**: Nhiều người có thể truy cập  
✅ **Remote access**: Chạy từ xa  
✅ **Database**: Lưu trữ lịch sử tốt hơn  
✅ **Responsive**: Tương thích mobile  
✅ **Auto refresh**: Cập nhật realtime  

## 🚀 Mở Rộng Tương Lai

- [ ] **Video streaming**: Hiển thị video realtime
- [ ] **User authentication**: Đăng nhập/đăng ký
- [ ] **Advanced analytics**: Biểu đồ chi tiết
- [ ] **Export data**: Xuất Excel/PDF
- [ ] **Multi-camera**: Xử lý nhiều camera
- [ ] **Real-time alerts**: Thông báo tắc đường
- [ ] **API**: REST API cho tích hợp

## 📞 Hỗ Trợ

Nếu có vấn đề, hãy kiểm tra:
1. Model file có tồn tại không
2. Dependencies đã cài đặt đầy đủ chưa
3. Port 5000 có bị chiếm không
4. Video format có được hỗ trợ không

---

