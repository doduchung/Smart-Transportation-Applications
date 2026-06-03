# 🚀 HƯỚNG DẪN CHẠY HỆ THỐNG ĐẾM PHƯƠNG TIỆN

## 📋 TỔNG QUAN

Hệ thống có **2 phiên bản** để bạn lựa chọn:

### 🖥️ **1. Desktop Python GUI** (Cửa sổ Python)
- **File**: `traffic_counter_improved.py`
- **Giao diện**: Cửa sổ Python với Tkinter
- **Tính năng**: Đầy đủ, mượt mà, dễ sử dụng
- **Phù hợp**: Sử dụng trực tiếp trên máy

### 🌐 **2. Web Application** (Trang web)
- **File**: `web_app_simple.py`
- **Giao diện**: Trang web trên trình duyệt
- **Tính năng**: Real-time streaming, database
- **Phù hợp**: Chia sẻ, truy cập từ xa

---

## 🖥️ CHẠY DESKTOP PYTHON GUI

### ⚡ **Cách 1: Chạy nhanh**
```bash
python traffic_counter_improved.py
```

### 📋 **Tính năng Desktop GUI**
- ✅ **Chọn video** từ máy tính
- ✅ **Bắt đầu/Dừng** xử lý
- ✅ **Vẽ vạch đếm** bằng chuột
- ✅ **Reset phiên** hiện tại
- ✅ **Debug mode** để kiểm tra
- ✅ **Lưu kết quả** vào file
- ✅ **Xem lịch sử** các phiên
- ✅ **Thống kê realtime**

### 🎮 **Hướng dẫn sử dụng Desktop**
1. **Chọn Video**: Nhấn "Chọn Video" → chọn file video
2. **Vẽ Vạch**: Nhấn "Kẻ Line" → click 2 điểm trên video
3. **Bắt đầu**: Nhấn "Bắt Đầu" để xử lý
4. **Xem kết quả**: Theo dõi số liệu realtime
5. **Lưu kết quả**: Nhấn "Lưu Kết Quả"

---

## 🌐 CHẠY WEB APPLICATION

### ⚡ **Cách 1: Chạy nhanh**
```bash
python web_app_simple.py
```

### 🌍 **Truy cập Web App**
- **URL**: `http://localhost:5001`
- **Giao diện**: Mở trình duyệt và truy cập URL

### 📋 **Tính năng Web App**
- ✅ **Upload video** qua web interface
- ✅ **Real-time video stream** (xem video đang xử lý)
- ✅ **Vẽ vạch đếm** trên web
- ✅ **Thống kê realtime** trong browser
- ✅ **Lịch sử** với database SQLite
- ✅ **Responsive design** (hoạt động trên mobile)

### 🎮 **Hướng dẫn sử dụng Web**
1. **Mở trình duyệt**: Truy cập `http://localhost:5001`
2. **Chọn video**: Nhập đường dẫn video
3. **Vẽ vạch**: Nhấn "Vẽ Vạch Đếm" → click 2 điểm
4. **Bắt đầu**: Nhấn "Bắt Đầu" để xử lý
5. **Xem realtime**: Video và thống kê cập nhật liên tục
6. **Lịch sử**: Xem trang "Lịch Sử"

---

## 🔧 SO SÁNH HAI PHIÊN BẢN

| Tính năng | Desktop GUI | Web App |
|-----------|-------------|---------|
| **Giao diện** | Cửa sổ Python | Trang web |
| **Video Stream** | Cửa sổ OpenCV | Browser |
| **Vẽ vạch** | Click trên video | Click trên web |
| **Database** | JSON file | SQLite |
| **Performance** | Cao | Trung bình |
| **Chia sẻ** | Khó | Dễ |
| **Mobile** | Không | Có |
| **Setup** | Đơn giản | Cần browser |

---

## 🚀 CHẠY CẢ HAI CÙNG LÚC

### 💡 **Lưu ý quan trọng**
- **Desktop GUI**: Chạy độc lập
- **Web App**: Chạy trên port 5001
- **Không xung đột**: Có thể chạy cả hai cùng lúc

### 🔄 **Workflow khuyến nghị**
1. **Development**: Sử dụng Desktop GUI để test nhanh
2. **Demo**: Sử dụng Web App để trình bày
3. **Production**: Chọn phiên bản phù hợp với nhu cầu

---

## 📊 CÔNG NGHỆ SỬ DỤNG

### 🤖 **AI & Computer Vision**
- **YOLOv8**: Object detection
- **OpenCV**: Video processing
- **Custom Model**: Train trên dataset Việt Nam

### 🖥️ **Desktop GUI**
- **Tkinter**: Python GUI framework
- **Threading**: Multi-threading processing
- **JSON**: Data storage

### 🌐 **Web Application**
- **Flask**: Python web framework
- **HTML/CSS/JS**: Frontend
- **SQLite**: Database
- **Bootstrap**: Responsive UI

---

## 🐛 TROUBLESHOOTING

### ❌ **Lỗi thường gặp**

#### **Desktop GUI**
```bash
# Lỗi: ModuleNotFoundError
pip install opencv-python ultralytics tkinter

# Lỗi: Model không load
# Kiểm tra file: traffic_training_small/traffic_yolov8n_small/weights/best.pt
```

#### **Web App**
```bash
# Lỗi: Port đã được sử dụng
# Thay đổi port trong web_app_simple.py:
app.run(host='0.0.0.0', port=5002, debug=True)

# Lỗi: Video không hiển thị
# Kiểm tra Console (F12) trong browser
```

### 🔍 **Debug Tips**
1. **Desktop**: Xem terminal output
2. **Web**: Mở Console (F12) trong browser
3. **Model**: Kiểm tra file best.pt có tồn tại
4. **Video**: Thử với video khác

---

## 📈 PERFORMANCE OPTIMIZATION

### ⚡ **Desktop GUI**
- **Detection interval**: 0.15s (có thể điều chỉnh)
- **Frame size**: 1280x720
- **Confidence**: 0.3

### 🌐 **Web App**
- **Update interval**: 150ms
- **JPEG quality**: 70%
- **Frame size**: 640x480

---

## 🎯 KẾT LUẬN

### 🏆 **Khuyến nghị**
- **Sử dụng thường xuyên**: Desktop GUI
- **Demo/trình bày**: Web App
- **Phát triển**: Cả hai để so sánh

### 🚀 **Bước tiếp theo**
1. **Test cả hai phiên bản**
2. **Chọn phiên bản phù hợp**
3. **Tùy chỉnh theo nhu cầu**
4. **Deploy production** (nếu cần)

---

**Chúc bạn sử dụng hiệu quả! 🎉**
