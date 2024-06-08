import numpy as np
import cv2


def get_data():
    # dán nhãn dữ liệu
    # lưu ký tự vào file data.npy và nhãn tương ứng vào file labels.npy
    import os
    from PIL import Image
    data = []
    labels = []
    nb_classes = 32

    for i in range(nb_classes):
        path = os.path.join("", '../data/characters', str(i))
        images = os.listdir(path)
        for a in images:
            try:
                image = Image.open(path + '/' + a).convert('L')
                image = image.resize((28, 28))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except (Exception, ):
                print("Error loading image")

    data = np.array(data)
    labels = np.array(labels)

    np.save('../data/data.npy', data)
    np.save('../data/labels.npy', labels)


def get_labels(path):
    # lấy nhãn cần nhận diện - NUMBER PLATE (trong file yolo.names)
    with open(path, 'r') as file:
        lines = file.readlines()

    return [line.strip() for line in lines]


def draw_labels_and_boxes(image, labels, boxes):
    # vẽ hộp giới hạn và biển số xe
    x_min = round(boxes[0])
    y_min = round(boxes[1])
    x_max = round(boxes[0] + boxes[2])
    y_max = round(boxes[1] + boxes[3])

    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)
    image = cv2.putText(image, labels, (x_min - 20, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.25, color=(0, 0, 255),
                        thickness=2)

    return image


def get_output_layers(model):
    layers_name = model.getLayerNames()  # lấy danh sách tên của tất cả các lớp trong mô hình
    # lấy vị trí của các lớp đầu ra trong danh sách
    output_layers = [layers_name[i - 1] for i in model.getUnconnectedOutLayers()]
    # trả về danh sách các lớp đầu ra
    return output_layers


def order_points(coordinates):
    # sắp xếp các tọa độ thành một mảng hình chữ nhật theo thứ tự nhất định
    # tạo một mảng numpy kích thước 4x2
    rect = np.zeros((4, 2), dtype="float32")
    # gán tọa độ điểm trên cùng bên trái, chiều rộng, chiều cao cho các biến
    x_min, y_min, width, height = coordinates

    # tọa độ điểm trên cùng bên trái
    rect[0] = np.array([round(x_min), round(y_min)])
    # tọa độ điểm trên cùng bên phải
    rect[1] = np.array([round(x_min + width), round(y_min)])
    # tọa độ điểm dưới cùng bên trái
    rect[2] = np.array([round(x_min), round(y_min + height)])
    # tọa độ điểm dưới cùng bên phải
    rect[3] = np.array([round(x_min + width), round(y_min + height)])

    return rect


def convert2_square(image):
    # """
    # chuyển đổi hình ảnh thành hình ảnh vuông bằng các đệm thêm các pixel ở xung quanh
    # :param image: input images
    # :return: numpy array
    # """

    img_h = image.shape[0]  # chiều cao ảnh đầu vào
    img_w = image.shape[1]  # chiều rộng ảnh đầu vào

    # if height > width
    if img_h > img_w:
        diff = img_h - img_w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = np.zeros(shape=(img_h, (diff//2) + 1))

        squared_image = np.concatenate((x1, image, x2), axis=1)
    elif img_w > img_h:
        diff = img_w - img_h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1

        squared_image = np.concatenate((x1, image, x2), axis=0)
    else:
        squared_image = image

    return squared_image


if __name__ == "__main__":
    get_data()
    print("\n-----Successful-----\n")
