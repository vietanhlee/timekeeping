from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

import cv2
from MainUi import Ui_MainWindow
from ultralytics import YOLO
model = YOLO(r'model/yolov11n-face.pt')

class HandlePageGetData(Ui_MainWindow):
    def __init__(self, MainWindow):
        super().__init__(MainWindow)
        self.stackedWidget.setCurrentWidget(self.page_get_data)
        self.mode_cam = 0 # 0 mặc định là start_detect
        # Khởi tạo camera
        self.cap = cv2.VideoCapture(0) 
        if not self.cap.isOpened():
            print("Không thể mở camera")
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # Start the camera feed
        self.timer.start(30)  # Update every 30ms (approx 33 FPS)

        self.push_get_data_face.clicked.connect(lambda : self.timer.timeout.connect(self.start_detect))
        self.push_stop_get_data.clicked.connect(lambda : self.timer.timeout.connect(self.update_frame))

    def start_detect(self):
        if(self.mode_cam == 0):
            self.timer.timeout.disconnect(self.update_frame)
            self.mode_cam = 1

        check_done, frame = self.cap.read()
        if(check_done == False):
            print('Đọc ảnh không thành công')
        # Đảo ảnh
        frame = cv2.flip(frame, 1)
        self.view_face.setPixmap(QPixmap.fromImage().scaled(self.view_face.size()))
    
    def update_frame(self):
        if(self.mode_cam == 1):
            self.timer.timeout.disconnect(self.start_detect)
            self.mode_cam = 0
        # Đọc một khung hình từ camera
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            # Chuyển đổi khung hình từ BGR (OpenCV) sang RGB (Qt)
            frame = self.convert_qimg(frame)
            # Hiển thị khung hình trên QLabel
            self.cam_view_main.setPixmap(QPixmap.fromImage(frame).scaled(self.cam_view_main.size()))

    def closeEvent(self, event):
        # Release the camera and stop the timer when the application closes
        if self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        event.accept()

    def convert_qimg(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        res = QImage(image.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        return res
    
if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = HandlePageGetData(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())