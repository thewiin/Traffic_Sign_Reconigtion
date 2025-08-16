import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = 'traffic_sign_model.h5' 
IMG_SIZE = 32

print("Đang tải mô hình...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Tải mô hình thành công!")
except Exception as e:
    print(f"Lỗi: Không thể tải mô hình. Vui lòng kiểm tra lại đường dẫn file '{MODEL_PATH}'.")
    print(f"Chi tiết lỗi: {e}")
    exit()

class_names = [
    'Tốc độ tối đa 20km/h',                           # 0
    'Tốc độ tối đa 30km/h',                           # 1
    'Tốc độ tối đa 50km/h',                           # 2
    'Tốc độ tối đa 60km/h',                           # 3
    'Tốc độ tối đa 70km/h',                           # 4
    'Tốc độ tối đa 80km/h',                           # 5
    'Hết hạn chế tốc độ 80km/h',                      # 6
    'Tốc độ tối đa 100km/h',                          # 7
    'Tốc độ tối đa 120km/h',                          # 8
    'Cấm vượt',                                      # 9
    'Cấm xe tải trên 3.5 tấn vượt',                   # 10
    'Giao nhau với đường ưu tiên',                     # 11
    'Đường ưu tiên',                                 # 12
    'Nhường đường',                                   # 13
    'Dừng lại',                                       # 14
    'Cấm xe cơ giới',                                # 15
    'Cấm xe tải trên 3.5 tấn',                        # 16
    'Cấm đi ngược chiều',                             # 17
    'Nguy hiểm khác',                                 # 18
    'Chỗ ngoặt nguy hiểm bên trái',                   # 19
    'Chỗ ngoặt nguy hiểm bên phải',                  # 20
    'Đường cong kép',                                 # 21
    'Đường gập ghềnh',                                # 22
    'Đường trơn',                                     # 23
    'Đường hẹp bên phải',                             # 24
    'Công trường',                                   # 25
    'Giao nhau có tín hiệu đèn',                       # 26
    'Đường người đi bộ cắt ngang',                    # 27
    'Trẻ em qua đường',                               # 28
    'Đường người đi xe đạp cắt ngang',                # 29
    'Cẩn thận băng tuyết',                            # 30
    'Thú rừng qua đường',                             # 31
    'Hết mọi lệnh cấm',                               # 32
    'Rẽ phải (hướng bắt buộc)',                       # 33
    'Rẽ trái (hướng bắt buộc)',                        # 34
    'Đi thẳng (hướng bắt buộc)',                      # 35
    'Đi thẳng hoặc rẽ phải',                          # 36
    'Đi thẳng hoặc rẽ trái',                           # 37
    'Đi bên phải',                                   # 38
    'Đi bên trái',                                    # 39
    'Bùng binh (vòng xuyến)',                         # 40
    'Hết cấm vượt',                                   # 41
    'Hết cấm xe tải trên 3.5 tấn vượt'                # 42
]

def predict_image(input_image):
    if input_image is None:
        return "Vui lòng tải ảnh lên."
    
    img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)[0]
    print("Dự đoán thô:", predictions) 
    
    confidences = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return confidences

print("\nĐang khởi tạo giao diện Gradio...")
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(label="Tải ảnh Biển báo Giao thông của bạn"),
    outputs=gr.Label(num_top_classes=3, label="Kết quả Dự đoán"),
    title="Demo Nhận dạng Biển báo Giao thông",
    description="Tải lên một bức ảnh biển báo giao thông để xem mô hình dự đoán. Kết quả sẽ hiển thị 3 loại biển báo có xác suất cao nhất với tên cụ thể.",
    examples=[]
)

print("\nGiao diện sẵn sàng! Hãy mở trình duyệt của bạn.")
iface.launch()