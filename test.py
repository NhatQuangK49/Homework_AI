import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from mtcnn import MTCNN

# Thư viện cho GUI
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QLineEdit, 
                             QMessageBox, QFrame, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QBrush
from PyQt5.QtCore import Qt, QSize

# --- Cấu hình giống như khi huấn luyện ---
DATA_DIR = 'dataset' # Cần để LabelEncoder khởi tạo đúng
IMG_SIZE = (96, 96) # Kích thước chuẩn hóa của khuôn mặt

# Khởi tạo MTCNN detector
detector = MTCNN()

# Hàm tiền xử lý ảnh (giống hệt trong file huấn luyện)
def preprocess_image(image_path):
    """
    Tải ảnh, phát hiện khuôn mặt, cắt, thay đổi kích thước và chuẩn hóa.
    Nếu tìm thấy khuôn mặt, trả về khuôn mặt đã xử lý.
    Nếu không, trả về None.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, "Không thể đọc ảnh."

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img_rgb)

    if detections:
        best_detection = max(detections, key=lambda x: x['confidence'])
        x, y, width, height = best_detection['box']
        
        # Mở rộng bounding box một chút để lấy thêm phần trán, cằm
        margin_x = int(width * 0.1)
        margin_y = int(height * 0.1)
        x_min = max(0, x - margin_x)
        y_min = max(0, y - margin_y)
        x_max = min(img.shape[1], x + width + margin_x)
        y_max = min(img.shape[0], y + height + margin_y)
        
        face_img = img_rgb[y_min:y_max, x_min:x_max]
        face_resized = cv2.resize(face_img, IMG_SIZE)
        face_normalized = face_resized.astype('float32') / 255.0
        return face_normalized, None
    else:
        return None, "Không tìm thấy khuôn mặt trong ảnh."

# Hàm dự đoán khuôn mặt
def predict_face(image_path, model, label_encoder, detector_func, img_size):
    """
    Dự đoán khuôn mặt trong một ảnh.
    """
    face_data, error_msg = detector_func(image_path)
    if face_data is not None:
        face_data = np.expand_dims(face_data, axis=0) # Thêm chiều batch
        
        prediction = model.predict(face_data)[0]
        
        predicted_class_idx = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = prediction[predicted_class_idx]
        
        return predicted_label, confidence, None
    return None, None, error_msg


class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.label_encoder = None
        self.current_image_path = None
        self.initUI()
        self.load_resources()

    def initUI(self):
        self.setWindowTitle('Hệ Thống Nhận Diện Khuôn Mặt (ANN)')
        self.setGeometry(100, 100, 900, 700) # (x, y, width, height)
        
        # --- Layout chính ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # --- Tiêu đề ứng dụng ---
        title_label = QLabel('ỨNG DỤNG NHẬN DIỆN KHUÔN MẶT CÁ NHÂN (ANN)')
        title_label.setFont(QFont('Segoe UI', 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 10px; background-color: #ecf0f1; border-radius: 8px;")
        main_layout.addWidget(title_label)

        # --- Phần chọn ảnh và đường dẫn ---
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Đường dẫn ảnh...")
        self.path_input.setFont(QFont('Segoe UI', 10))
        self.path_input.setStyleSheet("padding: 8px; border: 1px solid #bdc3c7; border-radius: 5px;")
        
        self.browse_button = QPushButton('Chọn Ảnh')
        self.browse_button.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.browse_button.setStyleSheet("background-color: #3498db; color: white; padding: 8px 15px; border-radius: 5px;")
        self.browse_button.clicked.connect(self.browse_image)
        
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_button)
        main_layout.addLayout(path_layout)

        # --- Phần hiển thị ảnh và kết quả ---
        content_layout = QHBoxLayout()

        # Left side: Image Display
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.StyledPanel)
        image_frame.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7; border-radius: 8px;")
        image_layout = QVBoxLayout()
        
        self.image_label = QLabel("Chưa có ảnh nào được chọn.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont('Segoe UI', 10, QFont.Weight.Light, True)) # Hoặc QFont.Weight.Normal
        self.image_label.setStyleSheet("color: #7f8c8d; padding: 20px;")
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.image_label)
        image_frame.setLayout(image_layout)
        
        content_layout.addWidget(image_frame, 3) # Chiếm 3 phần chiều rộng

        # Right side: Results
        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.StyledPanel)
        results_frame.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7; border-radius: 8px;")
        results_layout = QVBoxLayout()
        results_layout.setSpacing(10)

        results_title = QLabel('KẾT QUẢ NHẬN DIỆN')
        results_title.setFont(QFont('Segoe UI', 14, QFont.Bold))
        results_title.setAlignment(Qt.AlignCenter)
        results_title.setStyleSheet("color: #2c3e50; padding: 10px; border-bottom: 1px solid #bdc3c7;")
        results_layout.addWidget(results_title)

        self.result_label = QLabel('Chưa có kết quả.')
        self.result_label.setFont(QFont('Segoe UI', 12))
        self.result_label.setStyleSheet("color: #34495e; padding: 10px;")
        self.result_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.result_label)

        self.confidence_label = QLabel('Độ tin cậy: N/A')
        self.confidence_label.setFont(QFont('Segoe UI', 12))
        self.confidence_label.setStyleSheet("color: #34495e; padding: 5px;")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.confidence_label)
        
        # Spacer để đẩy nội dung lên trên
        results_layout.addStretch(1)

        self.recognize_button = QPushButton('Nhận Diện Khuôn Mặt')
        self.recognize_button.setFont(QFont('Segoe UI', 12, QFont.Bold))
        self.recognize_button.setStyleSheet("background-color: #2ecc71; color: white; padding: 12px 20px; border-radius: 8px;")
        self.recognize_button.clicked.connect(self.recognize_face)
        results_layout.addWidget(self.recognize_button)

        results_frame.setLayout(results_layout)
        content_layout.addWidget(results_frame, 2) # Chiếm 2 phần chiều rộng

        main_layout.addLayout(content_layout)
        
        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #f0f2f5;") # Màu nền tổng thể

    def load_resources(self):
        """Tải mô hình và khởi tạo LabelEncoder."""
        try:
            print("Đang tải mô hình...")
            self.model = load_model('face_recognition_ann_model.h5')
            print("Mô hình đã được tải thành công.")

            print("Đang chuẩn bị LabelEncoder...")
            if not os.path.exists(DATA_DIR):
                QMessageBox.critical(self, "Lỗi Tài Nguyên", 
                                     f"Thư mục '{DATA_DIR}' không tìm thấy. Không thể khởi tạo LabelEncoder chính xác.\n"
                                     "Vui lòng đảm bảo thư mục 'dataset' (chứa Hung, Quang, Uyen) nằm cùng cấp với test.py.")
                self.close()
                return

            class_names_from_dir = sorted(os.listdir(DATA_DIR))
            class_names_from_dir = [name for name in class_names_from_dir if os.path.isdir(os.path.join(DATA_DIR, name))]

            if not class_names_from_dir:
                QMessageBox.critical(self, "Lỗi Tài Nguyên", 
                                     f"Không tìm thấy thư mục con nào trong '{DATA_DIR}'.\n"
                                     "Vui lòng đảm bảo 'dataset' chứa các thư mục 'Hung', 'Quang', 'Uyen'.")
                self.close()
                return

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(class_names_from_dir)
            print(f"LabelEncoder đã được khởi tạo với các lớp: {list(self.label_encoder.classes_)}")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi Tải Tài Nguyên", f"Không thể tải mô hình hoặc khởi tạo LabelEncoder: {e}")
            self.close() # Đóng ứng dụng nếu có lỗi nghiêm trọng

    def browse_image(self):
        """Mở hộp thoại chọn ảnh và hiển thị ảnh."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn Ảnh", "", 
                                                   "Ảnh (*.png *.jpg *.jpeg);;Tất cả Files (*)", options=options)
        if file_path:
            self.current_image_path = file_path
            self.path_input.setText(file_path)
            self.display_image(file_path)
            self.result_label.setText('Chưa có kết quả.')
            self.confidence_label.setText('Độ tin cậy: N/A')

    def display_image(self, path):
        """Hiển thị ảnh được chọn trên QLabel."""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.image_label.setText("Không thể tải ảnh. Vui lòng chọn file ảnh hợp lệ.")
            self.image_label.setStyleSheet("color: red; padding: 20px;")
        else:
            # Scale pixmap để vừa với kích thước của QLabel
            # Giữ tỷ lệ khung hình
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setText("") # Xóa text nếu ảnh đã được hiển thị
            self.image_label.setStyleSheet("border: none;") # Xóa style lỗi nếu có

    def recognize_face(self):
        """Thực hiện nhận diện khuôn mặt khi nhấn nút."""
        if self.model is None or self.label_encoder is None:
            QMessageBox.warning(self, "Cảnh báo", "Mô hình hoặc bộ mã hóa nhãn chưa được tải. Vui lòng khởi động lại ứng dụng.")
            return

        if not self.current_image_path:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một ảnh trước khi nhận diện.")
            return

        print(f"Đang nhận diện khuôn mặt trong ảnh: {self.current_image_path}")
        label, confidence, error_msg = predict_face(self.current_image_path, self.model, self.label_encoder, preprocess_image, IMG_SIZE)

        if error_msg:
            self.result_label.setText(f"Lỗi: {error_msg}")
            self.result_label.setStyleSheet("color: red; font-weight: bold; padding: 10px;")
            self.confidence_label.setText('Độ tin cậy: N/A')
        elif label:
            self.result_label.setText(f"<b>Người:</b> {label}")
            self.confidence_label.setText(f"<b>Độ tin cậy:</b> {confidence*100:.2f}%")
            self.result_label.setStyleSheet("color: #27ae60; font-size: 16px; padding: 10px;")
            self.confidence_label.setStyleSheet("color: #2980b9; font-size: 14px; padding: 5px;")
            
            if confidence < 0.7: # Ngưỡng tin cậy thấp
                self.result_label.setText(f"<b>Người:</b> {label} (Độ tin cậy thấp!)")
                self.result_label.setStyleSheet("color: #e67e22; font-size: 16px; padding: 10px;")

        else:
            self.result_label.setText("Không tìm thấy khuôn mặt hoặc không thể nhận diện.")
            self.confidence_label.setText('Độ tin cậy: N/A')
            self.result_label.setStyleSheet("color: #c0392b; font-weight: bold; padding: 10px;")


if __name__ == '__main__':
    app = QApplication([])
    window = FaceRecognitionApp()
    window.show()
    app.exec_()