# Traffic Sign Recognition

## Giới thiệu
Dự án **Nhận diện Biển báo Giao thông (Traffic Sign Recognition)** sử dụng công nghệ Học sâu (Deep Learning) để phân loại các biển báo giao thông thành 43 loại khác nhau, dựa trên bộ dữ liệu GTSRB. 

Dự án bao gồm mã nguồn để huấn luyện một mô hình mạng nơ-ron tích chập (CNN) có tích hợp Squeeze-and-Excitation (SE block), và một ứng dụng web nhỏ sử dụng Gradio giúp người dùng dễ dàng tải ảnh lên để nhận diện thử.

## Các tính năng chính
- **Tự động tải bộ dữ liệu**: Sử dụng thư viện `deeplake` để tải bộ dữ liệu GTSRB nhanh chóng.
- **Phân tích dữ liệu trực quan (EDA)**: Vẽ biểu đồ phân bố lớp và hiển thị các ảnh mẫu từ bộ dữ liệu.
- **Mô hình CNN tối ưu**: Sử dụng `SeparableConv2D` giúp giảm số lượng tham số tính toán, kết hợp với khối `Squeeze-and-Excitation` giúp cải thiện độ chính xác.
- **Tăng cường dữ liệu (Data Augmentation)**: Áp dụng xoay, thu phóng và dịch chuyển ảnh để mô hình tổng quát hóa tốt hơn.
- **Giao diện web trực quan**: Ứng dụng demo với `Gradio` cho phép nhận diện biển báo trực tiếp từ ảnh tải lên.

## Yêu cầu hệ thống (Dependencies)
Dự án yêu cầu môi trường Python 3.7 trở lên. Các thư viện cần thiết bao gồm:
- `tensorflow`
- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `deeplake`
- `gradio`

## Hướng dẫn cài đặt
1. Mở terminal/command prompt và di chuyển vào thư mục dự án:
   ```bash
   cd Traffic_Sign_Reconigtion
   ```

2. Cài đặt các thư viện cần thiết bằng lệnh pip:
   ```bash
   pip install tensorflow numpy opencv-python matplotlib scikit-learn tqdm deeplake gradio
   ```

## Cách sử dụng

### 1. Huấn luyện mô hình (Training)
Nếu bạn muốn tự huấn luyện lại mô hình, hãy chạy file `main.py`:
```bash
python main.py
```
**Quá trình này sẽ thực hiện:**
- Tải dữ liệu GTSRB thông qua `deeplake`.
- Thực hiện khám phá dữ liệu (hiển thị biểu đồ phân bố và ảnh mẫu).
- Bắt đầu quá trình huấn luyện mô hình với 25 Epochs (có Early Stopping để dừng sớm nếu mô hình không cải thiện).
- Vẽ và hiển thị ma trận nhầm lẫn (Confusion Matrix) cùng với các biểu đồ về độ chính xác và mất mát (Loss/Accuracy plots).
- Lưu lại trọng số mô hình tốt nhất vào file `traffic_sign_model.h5`.

### 2. Chạy Demo Giao diện Web (Inference)
Nếu dự án đã có sẵn file mô hình `traffic_sign_model.h5`, bạn có thể chạy ngay ứng dụng web để nhận diện ảnh biển báo:
```bash
python demo.py
```
- Khi chạy xong lệnh trên, terminal sẽ xuất ra một đường dẫn (ví dụ: `http://127.0.0.1:7860/`). 
- Mở đường dẫn đó trên trình duyệt web của bạn, tải lên một bức ảnh biển báo bất kỳ và mô hình sẽ trả về 3 kết quả dự đoán có xác suất cao nhất.

