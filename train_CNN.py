import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mtcnn import MTCNN # Thư viện phát hiện khuôn mặt MTCNN
import matplotlib.pyplot as plt

# Đường dẫn đến thư mục chứa dữ liệu
DATA_DIR = 'Dataset'
IMG_SIZE = (64, 64) # Kích thước chuẩn hóa của khuôn mặt

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
        # face = face.astype('float32') / 255.0 # Chuẩn hóa pixel, sẽ được thực hiện bởi ImageDataGenerator
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

print(f"Tổng số ảnh khuôn mặt gốc đã xử lý: {len(X)}")
print(f"Số lượng lớp: {len(class_names)}")
print(f"Tên các lớp: {class_names}")

# Mã hóa nhãn văn bản thành số nguyên
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(class_names)

# Mã hóa nhãn thành one-hot encoding
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
# Sử dụng stratify để đảm bảo tỷ lệ lớp được giữ nguyên
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"Kích thước tập huấn luyện ban đầu: {X_train.shape}, {y_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}, {y_test.shape}")

# Bước 2: Cấu hình Tăng cường dữ liệu (Data Augmentation)
# Tăng cường gấp 20 lần số ảnh, nên ta sẽ tính tổng số ảnh cần tạo
# Mỗi epoch, ImageDataGenerator sẽ tạo ra các biến thể mới từ X_train
# Dưới đây là các tham số tăng cường mà bạn có thể điều chỉnh
datagen = ImageDataGenerator(
    rotation_range=20,          # Xoay ảnh ngẫu nhiên tối đa 20 độ
    width_shift_range=0.2,      # Dịch chuyển chiều rộng ngẫu nhiên
    height_shift_range=0.2,     # Dịch chuyển chiều cao ngẫu nhiên
    shear_range=0.2,            # Cắt xén ảnh
    zoom_range=0.2,             # Phóng to/thu nhỏ ngẫu nhiên
    horizontal_flip=True,       # Lật ảnh theo chiều ngang ngẫu nhiên
    brightness_range=[0.8, 1.2],# Thay đổi độ sáng ngẫu nhiên
    rescale=1./255              # Chuẩn hóa pixel về [0, 1]
)

# Để tăng cường dữ liệu gấp 20 lần, chúng ta sẽ huấn luyện với số bước (steps_per_epoch) lớn hơn
# Một cách đơn giản để ước tính steps_per_epoch là:
# (số ảnh huấn luyện ban đầu * hệ số tăng cường) // batch_size
# Tuy nhiên, ImageDataGenerator sẽ tự động tạo vô số ảnh khi bạn gọi flow hoặc flow_from_directory.
# Quan trọng là số epoch và steps_per_epoch.
# Giả sử chúng ta muốn mỗi epoch tương đương với việc "nhìn thấy" 20 lần số ảnh gốc
original_train_samples = X_train.shape[0]
augmentation_factor = 5
batch_size = 32
steps_per_epoch = (original_train_samples * augmentation_factor) // batch_size
print(f"Số bước mỗi epoch (để tăng cường dữ liệu x{augmentation_factor} lần): {steps_per_epoch}")

# Bước 3: Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25), # Thêm Dropout để tránh overfitting

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(), # Làm phẳng các đặc trưng từ Conv/Pool layers
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') # Lớp đầu ra với softmax cho phân loại
])

# Bước 4: Huấn luyện mô hình với Data Augmentation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\nBắt đầu huấn luyện mô hình CNN với tăng cường dữ liệu...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=35, # Bạn có thể tăng số epochs nếu cần
    validation_data=(X_test / 255.0, y_test) # Chuẩn hóa tập test
)

# Bước 5: Đánh giá mô hình
print("\nĐánh giá mô hình trên tập kiểm tra...")
loss, accuracy = model.evaluate(X_test / 255.0, y_test, verbose=0)
print(f"Độ chính xác trên tập kiểm tra: {accuracy*100:.2f}%")

# Lưu mô hình
model.save('nhan_dien_mat_CNN.h5')
print("Mô hình đã được lưu vào 'nhan_dien_mat_CNN.h5'")

# Hiển thị đồ thị huấn luyện
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
