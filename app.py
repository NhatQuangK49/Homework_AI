import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from mtcnn import MTCNN # Thư viện phát hiện khuôn mặt MTCNN

# Đường dẫn đến thư mục chứa dữ liệu
DATA_DIR = 'Dataset'
IMG_SIZE = (96, 96) # Kích thước chuẩn hóa của khuôn mặt

# Khởi tạo MTCNN detector
detector = MTCNN()

def preprocess_image(image_path):
    """
    Tải ảnh, phát hiện khuôn mặt, cắt, thay đổi kích thước và chuẩn hóa.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img_rgb)

    if detections:
        # Lấy khuôn mặt có độ tin cậy cao nhất
        best_detection = max(detections, key=lambda x: x['confidence'])
        x, y, width, height = best_detection['box']
        
        # Mở rộng bounding box một chút để lấy thêm phần trán, cằm
        margin_x = int(width * 0.1)
        margin_y = int(height * 0.1)
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        width = min(img.shape[1] - x, width + 2 * margin_x)
        height = min(img.shape[0] - y, height + 2 * margin_y)
        
        face = img_rgb[y:y+height, x:x+width]
        face = cv2.resize(face, IMG_SIZE)
        face = face.astype('float32') / 255.0 # Chuẩn hóa pixel
        return face
    else:
        # print(f"Không tìm thấy khuôn mặt trong ảnh: {image_path}")
        return None

def load_data(data_dir):
    """
    Tải và tiền xử lý dữ liệu từ các thư mục.
    """
    faces = []
    labels = []
    class_names = sorted(os.listdir(data_dir))

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            print(f"Đang xử lý thư mục: {class_name}")
            for filename in os.listdir(class_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(class_path, filename)
                    face = preprocess_image(image_path)
                    if face is not None:
                        faces.append(face)
                        labels.append(class_name)
    return np.array(faces), np.array(labels), class_names

# Bước 1: Chuẩn bị dữ liệu
print("Bắt đầu tải và tiền xử lý dữ liệu...")
X, y, class_names = load_data(DATA_DIR)

if len(X) == 0:
    print("Không tìm thấy dữ liệu khuôn mặt hợp lệ. Vui lòng kiểm tra lại thư mục và ảnh.")
    exit()

print(f"Tổng số ảnh khuôn mặt đã xử lý: {len(X)}")
print(f"Số lượng lớp: {len(class_names)}")
print(f"Tên các lớp: {class_names}")

# Mã hóa nhãn văn bản thành số nguyên
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(class_names)

# Mã hóa nhãn thành one-hot encoding
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"Kích thước tập huấn luyện: {X_train.shape}, {y_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}, {y_test.shape}")

# Bước 2: Xây dựng mô hình ANN
model = Sequential([
    Flatten(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)), # Làm phẳng ảnh 3D thành 1D
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax') # Lớp đầu ra với softmax cho phân loại
])

# Bước 3: Huấn luyện mô hình
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\nBắt đầu huấn luyện mô hình...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Bước 4: Đánh giá mô hình
print("\nĐánh giá mô hình trên tập kiểm tra...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Độ chính xác trên tập kiểm tra: {accuracy*100:.2f}%")

# Lưu mô hình (tùy chọn)
model.save('face_recognition_ann_model.h5')
print("Mô hình đã được lưu vào 'face_recognition_ann_model.h5'")

# Để kiểm tra một ảnh mới:
# from tensorflow.keras.models import load_model
# loaded_model = load_model('face_recognition_ann_model.h5')
#
# def predict_face(image_path, model, label_encoder):
#     face = preprocess_image(image_path)
#     if face is not None:
#         face = np.expand_dims(face, axis=0) # Thêm chiều batch
#         prediction = model.predict(face)[0]
#         predicted_class_idx = np.argmax(prediction)
#         predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
#         confidence = prediction[predicted_class_idx]
#         return predicted_label, confidence
#     return None, None
#
# # Ví dụ sử dụng:
# # test_image_path = 'path/to/your/new_image.jpg'
# # label, conf = predict_face(test_image_path, model, label_encoder)
# # if label:
# #     print(f"Khuôn mặt được nhận diện là: {label} với độ tin cậy: {conf*100:.2f}%")
# # else:
# #     print("Không thể nhận diện khuôn mặt.")

# Hiển thị đồ thị huấn luyện (tùy chọn)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()