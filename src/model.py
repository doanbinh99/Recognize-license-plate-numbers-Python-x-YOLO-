import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import *


class Model(object):
    def __init__(self):
        self.build_model()
        self.model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['acc'])

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='softmax'),
        ])

    def train(self):
        data = np.load('data/data.npy')
        labels = np.load('data/labels.npy')
        nb_classes = 32
        batch_size = 128

        # chia dữ liệu: 90% cho huấn luyện và 10% cho kiểm thử
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
        # chuyển đổi kiu dữ liệu thành kiểu float32 - kiểu dữ liệu thường được yêu cầu cho các mô hình học sâu
        x_train = X_train.astype('float32')
        x_test = X_test.astype('float32')
        # chuẩn hóa dữ liệu => cải thiển độ hội tụ và hiệu suất mô hình
        x_train /= 255
        x_test /= 255
        # định dạng lại dữ liệu đầu vào thành 28x28, 1 kênh màu xám
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        # mã hóa nhãn phân loại thành vector one-hot
        # nb_classes chỉ định số lớp nhãn
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        # giảm tốc độ học khi độ chính xác không được cải thiện
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1,)
        # lưu trọng số trong quá trình huấn luyện
        # cho phép đào tạo từ một điểm trước đó nếu quá trình huấn luyện bị gián đoạn
        cpt_save = tf.keras.callbacks.ModelCheckpoint('./weights/TempWeight.h5', save_best_only=True,
                                                      monitor='val_acc', mode='max')

        # huấn luyện mô hình
        history = self.model.fit(
            x_train, y_train,  # tập dữ liệu huấn luyện
            validation_split=0.1,  # dữ liệu kiểm thử lấy 10% từ tập huấn luyện
            callbacks=[cpt_save, reduce_lr],  # danh sách hàm gọi lại
            verbose=1,  # Mức độ chi tiết của đầu ra, 0: không hiển thị, 1: đơn giản, 2: chi tiết
            epochs=50,  # 50 vòng lặp
            shuffle=True,  # xáo trộn dữ liệu trước khi huấn luyện => đảm bảo hiệu suất, khả năng khái quát
            batch_size=batch_size  # kích thước 1 batch trong quá trình huấn luyện.
        )

        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Accuracy: ', score[1])
        print('Loss: ', score[0])

        self.model.save('./weights/model.h5')
        return history
