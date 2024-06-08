import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from src.lp_recognition import E2E


class LicensePlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('License Plate Recognition')
        self.frame = tk.Frame(root, padx=5, pady=5, highlightbackground="#000",
                              highlightthickness=1, background="#f5bd6e")
        self.frame.pack(expand=True, padx=5, pady=5)

        self.label = tk.Label(self.frame, text='NHẬN DIỆN BIỂN SỐ XE', foreground="white",
                              font=('Courier bold', 20), background="#f5bd6e")
        self.label.pack(padx=7, pady=7)

        self.button = tk.Button(self.frame, text='Chọn Ảnh', font=('Courier bold', 13),
                                activebackground='green', command=self.load_image)
        self.button.pack(padx=5, pady=5)

        # Tạo canvas và thiết lập kích thước cố định
        self.width = 500
        self.height = 500
        self.canvas = tk.Canvas(self.frame, width=self.width, height=self.height, background="white")
        self.canvas.pack()

        self.model = E2E()  # Tạo một đối tượng mô hình nhận diện biển số xe

        self.result_label = tk.Label(self.frame, text='', background="#f5bd6e")
        self.result_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            processed_image = self.model.predict(image)

            # Resize hình ảnh để vừa với canvas
            processed_image = self.resize_image(processed_image, self.width, self.height)

            # Chuyển đổi hình ảnh đã xử lý thành định dạng RGB để hiển thị với Tkinter
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            processed_image_pil = Image.fromarray(processed_image_rgb)
            processed_image_tk = ImageTk.PhotoImage(processed_image_pil)

            # Hiển thị hình ảnh lên canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=processed_image_tk)
            self.canvas.image = processed_image_tk

            # Nhận diện biển số và hiển thị kết quả
            license_plate = self.model.format()  # Truyền hình ảnh đã xử lý vào phương thức format()
            if license_plate:
                self.result_label.config(text=f'Biển số xe: {license_plate}', font=("Courier bold", 24), background="#fff")
                self.result_label.pack()
            else:
                self.result_label.pack_forget()

    def resize_image(self, image, width, height):
        return cv2.resize(image, (width, height))


if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    app = LicensePlateRecognitionApp(root)
    root.mainloop()
