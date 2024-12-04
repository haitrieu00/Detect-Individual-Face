import os
import cv2
import numpy as np
import tkinter as tk
from mtcnn import MTCNN
from tkinter import simpledialog, messagebox
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ============================
# Chụp và lưu khuôn mặt
# ============================
from mtcnn import MTCNN  # Thêm thư viện MTCNN

def capture_faces():
    name = simpledialog.askstring("Chụp Hình", "Nhập tên của người cần chụp:")
    if not name:
        messagebox.showerror("Lỗi", "Tên không được để trống.")
        return
    
    save_dir = "dataset_face"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    person_dir = os.path.join(save_dir, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    detector = MTCNN()  # Khởi tạo MTCNN
    cap = cv2.VideoCapture(0)
    count = 0
    num_images = 500
    
    print("Đang chụp và lưu khuôn mặt. Nhấn 'q' để thoát.")
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Không thể truy cập camera.")
            break
        
        results = detector.detect_faces(frame)  # Phát hiện khuôn mặt
        
        for result in results:
            if result['confidence'] > 0.9:  # Ngưỡng tự tin
                x, y, w, h = result['box']
                x, y = max(0, x), max(0, y)  # Đảm bảo không vượt ngoài khung hình
                face = frame[y:y+h, x:x+w]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(face_gray, (128, 128))
                
                # Lưu ảnh
                file_path = os.path.join(person_dir, f"{name}_{count}.jpg")
                cv2.imwrite(file_path, resized_face)
                
                count += 1
                print(f"Đã lưu khuôn mặt {count}/{num_images} vào {file_path}.")
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Chụp Khuôn Mặt", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Thông Báo", f"Đã chụp xong {count} hình ảnh cho {name}.")

# ============================
# Huấn luyện mô hình
# ============================
def load_and_train_model():
    def load_images(data_dir="dataset_face"):
        X, y = [], []
        for person_name in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person_name)
            if os.path.isdir(person_path):
                for img_name in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img_resized = cv2.resize(img, (128, 128)) / 255.0  # Chuẩn hóa
                        X.append(img_resized)
                        y.append(person_name)
        return np.array(X), np.array(y)
    
    X, y = load_images()
    X = X.reshape(-1, 128, 128, 1)  # Thêm chiều cho ảnh grayscale
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y)  # One-hot encoding
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    
    # Xây dựng mô hình CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  # Giảm overfitting
        
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(len(le.classes_), activation="softmax")
    ])
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # EarlyStopping and ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    
    model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        validation_data=(X_test, y_test),
        epochs=20,
        callbacks=[early_stopping, reduce_lr]
    )
    
    model.save("face_recognition_model.h5")
    messagebox.showinfo("Thông Báo", "Đã hoàn tất huấn luyện và lưu mô hình!")

# ============================
# Nhận diện khuôn mặt
# ============================
def recognize_faces():
    if not os.path.exists("face_recognition_model.h5"):
        messagebox.showerror("Lỗi", "Mô hình chưa được huấn luyện. Vui lòng huấn luyện trước!")
        return
    
    model = load_model("face_recognition_model.h5")
    detector = MTCNN()  # Khởi tạo MTCNN
    cap = cv2.VideoCapture(0)

    # Load LabelEncoder
    le = LabelEncoder()
    le.fit(os.listdir("dataset_face"))  # Dựa vào tên thư mục để mã hóa tên người
    
    print("Nhận diện khuôn mặt, nhấn 'q' để thoát.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể truy cập camera.")
            break
        
        results = detector.detect_faces(frame)  # Phát hiện khuôn mặt
        
        for result in results:
            if result['confidence'] > 0.9:  # Ngưỡng tự tin
                x, y, w, h = result['box']
                x, y = max(0, x), max(0, y)  # Đảm bảo không vượt ngoài khung hình
                face = frame[y:y+h, x:x+w]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(face_gray, (128, 128)) / 255.0
                input_face = np.expand_dims(resized_face, axis=(0, -1))
                
                # Dự đoán khuôn mặt
                prediction = model.predict(input_face)
                label_index = np.argmax(prediction)
                confidence = prediction[0][label_index]
                
                recognized_name = "Unknown"
                if confidence > 0.9:  # Ngưỡng nhận diện
                    recognized_name = le.inverse_transform([label_index])[0]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{recognized_name} ({confidence*100:.1f}%)", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Detect Face - MTCNN", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ============================
# Giao diện Tkinter
# ============================
def create_gui():
    root = tk.Tk()
    root.title("Nhận Diện Khuôn Mặt")
    root.geometry("300x300")
    
    tk.Button(root, text="Chụp Ảnh", command=capture_faces, width=20, height=2, bg="lightblue").pack(pady=20)
    tk.Button(root, text="Huấn Luyện", command=load_and_train_model, width=20, height=2, bg="lightyellow").pack(pady=20)
    tk.Button(root, text="Nhận Diện", command=recognize_faces, width=20, height=2, bg="lightgreen").pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    create_gui()
