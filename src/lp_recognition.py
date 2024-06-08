import cv2
import numpy as np
from skimage import measure
from imutils import perspective
import imutils

from src.data_utils import order_points, convert2_square, draw_labels_and_boxes
from src.lp_detection.detect import DetectNumberPlate
from skimage.filters import threshold_local
from src.model import Model

# dictionary chứa ký tự và nhãn cần nhận diện
ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

# đường dẫn tới file chứa trọng số, nhãn, cấu hình của mô hình YOLO
LP_DETECTION_CFG = {
    "weight_path": "./src/lp_detection/cfg/yolov3-tiny_15000.weights",
    "classes_path": "./src/lp_detection/cfg/yolo.names",
    "config_path": "./src/lp_detection/cfg/yolov3-tiny.cfg"
}

# đường dẫn tới file chứa trọng số mô hình CNN
CHAR_CLASSIFICATION_WEIGHTS = "E:\\Python\\license-plate-recognize\\weights\\model.h5"


class E2E(object):
    def __init__(self):
        self.image = np.empty((28, 28, 1))  # tạo một mảng numpy rỗng kích thước 28x28x1
        # khởi tạo đối tượng nhận diện biển số trong ảnh
        self.detectLP = DetectNumberPlate(LP_DETECTION_CFG['classes_path'], LP_DETECTION_CFG['config_path'],
                                          LP_DETECTION_CFG['weight_path'])
        self.recogChar = Model().model  # khởi tạo mô hình CNN
        self.recogChar.load_weights(CHAR_CLASSIFICATION_WEIGHTS)  # load trọng số mô hình CNN
        self.candidates = []  # danh sách rỗng dùng chứa hộp giới hạn

    def extract_lp(self):
        coordinates = self.detectLP.detect(self.image)  # lưu giá trị hộp giới hạn vào biến
        if len(coordinates) == 0:
            ValueError('No images detected')

        # tạo một generator trả về các tọa độ
        for coordinate in coordinates:
            yield coordinate

    def predict(self, image):
        # ảnh đầu vào
        self.image = image

        # khoanh vùng biển số xe
        for coordinate in self.extract_lp():
            self.candidates = []

            pts = order_points(coordinate)  # tạo độ 4 đỉnh của hộp giới hạn

            # biến đổi vùng biển số thành một hình chữ nhật vuông => dễ dàng nhận diện
            lp_region = perspective.four_point_transform(self.image, pts)
            # phân đoạn ký tự
            self.segmentation(lp_region)
            # nhận diện ký tự
            self.recognize_char()
            # lấy nhãn dự đoán
            license_plate = self.format()
            # vẽ hộp giới hạn và nhãn dự đoán lên hình ảnh
            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)

        return self.image

    def segmentation(self, lp_region):
        # lấy giá trị độ sáng của ảnh
        v = cv2.split(cv2.cvtColor(lp_region, cv2.COLOR_BGR2HSV))[2]

        # áp dụng ngưỡng thích nghi để tách nền và ký tự
        t = threshold_local(v, 15, offset=10, method="gaussian")
        # tạo ảnh nhị phân từ kênh V đã được áp dụng ngưỡng (ký tự đen, nền trắng)
        thresh = (v > t).astype("uint8") * 255

        # Đảo ngược màu ảnh bị phân (ký tự trắng, nền đen)
        thresh = cv2.bitwise_not(thresh)
        # thay đổi kích thước chiều rộng thành 400 pixels
        thresh = imutils.resize(thresh, width=400)
        # làm mờ ảnh để giảm nhiễu
        thresh = cv2.medianBlur(thresh, 5)

        # gán nhãn cho các vùng liên thông trong ảnh
        labels = measure.label(thresh, connectivity=2, background=0)

        # Xử lý từng thành phần liên thông
        for label in np.unique(labels):
            # bỏ qua nhãn nền
            if label == 0:
                continue

            # tạo một mặt nạ nhị phân
            mask = np.zeros(thresh.shape, dtype="uint8")
            # tô nền trắng lên mặt nạ cho các pixel thuộc thành phần liên thông hiện tại
            # thành phần liên thông là một vùng các pixel liền kề với nhau có cùng giá trị
            # tô trắng các phần này giúp phân tách ký tự
            mask[labels == label] = 255

            # tìm các đường bao của thành phần liên thông trong mặt nạ
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # tìm đường bao có diện tích lớn nhất
                contour = max(contours, key=cv2.contourArea)
                # tìm hình chữ nhật bao quanh đường bao lớn nhất, hcn này dùng để trích xuất ký tự
                (x, y, w, h) = cv2.boundingRect(contour)

                # tỉ lệ khung hình của hình chữ nhật => lọc ra các ký tự có hình dạng phù hợp
                aspect_ratio = w / float(h)
                # độ đặc (độ kín) của đường bao => lọc ra các ký tự không bị khuyết
                solidity = cv2.contourArea(contour) / float(w * h)
                # tỉ lệ chiều cao của hcn và toàn bộ vùng biển số => lọc ra các ký tự có kích thước phù hợp
                height_ratio = h / float(lp_region.shape[0])

                if 0.1 < aspect_ratio < 1.0 and solidity > 0.1 and 0.35 < height_ratio < 2.0:
                    # trích xuất vùng ký tự từ ảnh nhị phân dựa trên tọa độ và kích thước hcn bao quanh
                    candidate = np.array(mask[y:y + h, x:x + w])
                    # chuyển vùng ký tự thành hình vuông
                    square_candidate = convert2_square(candidate)
                    # thay hình dạng vùng ký tự thành một mảng 3 chiều với kích thước 28x28x1
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    # thêm vùng ký tự đã xử lý và tọa độ của nó vào danh sách
                    self.candidates.append((square_candidate, (y, x)))

    def recognize_char(self):
        characters = []  # dùng để chứa vùng ký tự
        coordinates = []  # dùng để chứa tọa độ ký tự

        for char, coordinate in self.candidates:
            characters.append(char)
            coordinates.append(coordinate)

        characters = np.array(characters)  # chuyển list thành numpy array
        result = self.recogChar.predict_on_batch(characters)  # dự đoán ký tự
        result_idx = np.argmax(result, axis=1)  # lưu chỉ số của ký tự được dự đoán

        self.candidates = []  # xóa danh sách cũ, để lưu trữ kết quả nhận diện
        for i in range(len(result_idx)):
            if result_idx[i] == 31:  # bỏ qua nhiễu, đường viền
                continue
            # lưu ký tự và tọa độ của nó
            self.candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

    def format(self):
        first_line = []  # lưu ký tự dòng trên
        second_line = []  # lưu ký tự dòng dưới

        for candidate, coordinate in self.candidates:
            # nếu ký tự nằm trong phạm vi 40 pixel bên trái so với ký tự đầu tiên => thuộc dòng trên
            if self.candidates[0][1][0] + 40 > coordinate[0]:
                first_line.append((candidate, coordinate[1]))
            else:
                second_line.append((candidate, coordinate[1]))

        def take_second(s):
            return s[1]

        # sắp xếp các ký tự mỗi dòng theo thức tự tăng dần tọa độ y
        first_line = sorted(first_line, key=take_second)
        second_line = sorted(second_line, key=take_second)

        if len(second_line) == 0:  # nếu biển số có 1 dòng => nối các ký tự first_line thành 1 chuỗi
            license_plate = "".join([str(ele[0]) for ele in first_line])
        else:   # nối các ký tự first_line và second_line thành một chuỗi
            license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0])
                                                                                          for ele in second_line])
        return license_plate
