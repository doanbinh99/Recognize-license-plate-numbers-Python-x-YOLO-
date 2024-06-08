import src.data_utils as utils
import cv2
import numpy as np


class DetectNumberPlate(object):
    def __init__(self, classes_path, config_path, weight_path, threshold=0.5):
        self.weight_path = weight_path
        self.cfg_path = config_path
        self.labels = utils.get_labels(classes_path)
        self.threshold = threshold  # ngưỡng tin cậy

        # đọc mạng YOLOv3 từ tham số weight và config_path
        self.model = cv2.dnn.readNet(model=self.weight_path, config=self.cfg_path)

    def detect(self, image):
        boxes = []
        classes_id = []
        confidences = []
        scale = 0.00392  # 1/255

        # chuyển ảnh đầu vào thành một blob
        blob = cv2.dnn.blobFromImage(image, scalefactor=scale, size=(416, 416), mean=(0, 0), swapRB=True, crop=False)
        height, width = image.shape[:2]

        # take image to model
        self.model.setInput(blob)

        # run forward
        # chạy lan truyền tiến và lấy kết quả từ các lớp
        outputs = self.model.forward(utils.get_output_layers(self.model))

        for output in outputs:
            for i in range(len(output)):
                scores = output[i][5:]  # trích xuất các điểm số cho mỗi lớp
                class_id = np.argmax(scores)  # tìm lớp có điểm số độ tin cậy cao nhất
                confidence = float(scores[class_id])  # lấy điểm số của lớp có độ tin cậy cao nhất

                if confidence > self.threshold:
                    # tọa độ điểm trung tâm
                    center_x = int(output[i][0] * width)
                    center_y = int(output[i][1] * height)
                    # chiều rộng, chiều cao
                    detected_width = int(output[i][2] * width)
                    detected_height = int(output[i][3] * height)
                    # tọa độ điểm trên cùng bên trái
                    x_min = center_x - detected_width / 2
                    y_min = center_y - detected_height / 2
                    # thêm tọa độ, id, độ tin cậy vào các list
                    boxes.append([x_min, y_min, detected_width, detected_height])
                    classes_id.append(class_id)
                    confidences.append(confidence)

        # lọc các khung giới hạn chồng lấn nhau, trả về chỉ số các hộp bao còn lại sau NMS
        # ngưỡng độ tin cậy tối thiểu là 0.5, hộp giới hạn có độ tin cậy nhỏ hơn sẽ bị loại bỏ
        # nms_threshold là ngưỡng IoU, 2 hộp bị coi là trùng nhau nếu chúng có IoU > 0.4
        # ngưỡng IoU = diện tích giao nhau / tổng diện tích 2 hình
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=self.threshold, nms_threshold=0.4)
        #lặp qua từng hộp còn lại sau NMS
        coordinates = []
        for i in indices:
            index = i
            x_min, y_min, width, height = boxes[index]
            x_min = round(x_min)
            y_min = round(y_min)

            coordinates.append((x_min, y_min, width, height))
        # trả về giá trị của danh sách các khung gới hạn sau khi lọc
        return coordinates
