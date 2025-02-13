from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
import sys
from handle_page_get_data import HandlePageGetData
from handle_page_train import HandelPageTrain
from handle_page_run import HandlePageRun
import cv2
import pickle
import numpy as np
from keras.api.models import load_model

class HandleMain(HandlePageGetData, HandelPageTrain, HandlePageRun):
    def __init__(self, MainWindow):
        HandlePageRun.__init__(self, MainWindow)
        HandelPageTrain.__init__(self, MainWindow)
        HandlePageGetData.__init__(self, MainWindow)
        
        self.log_res.setWordWrap(True)  # Cho phép tự động xuống dòng

        # Sử dụng layout để mở rộng tự động
        self.layout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.layout.addWidget(self.log_res)

        # Cần phải định nghĩa lại một số thuộc tính về các nút nhất kết nối với hàm nào 
        # Vì khi gọi init() lần lượt lên các class bố thì class bố cũng init() lên class ông 
        # Nên vô tình đã định nghĩa các thuộc tính dưới là None (không kết nối với cái gì cả)
        self.push_run.clicked.connect(lambda : self.timerr.timeout.connect(self.start_predict))
        self.push_stop.clicked.connect(lambda : self.timerr.timeout.connect(self.update_frame_run))
        self.push_get_data_face.clicked.connect(lambda : self.timer.timeout.connect(self.start_detect))
        self.push_stop_get_data.clicked.connect(lambda : self.timer.timeout.connect(self.update_frame))
        self.push_training.clicked.connect(self.start_training)
        self.push_check_in.clicked.connect(self.check_in)
        self.push_check_out.clicked.connect(self.check_out)
        
        self.radioButton.click()
        # Thiết đặt các kết nối khi nhấn các nút chuyển trang
        self.button_instrucst.clicked.connect(lambda: self.change_page(1))
        self.button_get_faces.clicked.connect(lambda: self.change_page(2))
        self.button_preprocess_training.clicked.connect(lambda: self.change_page(3))
        self.button_run_trial.clicked.connect(lambda: self.change_page(4))
        
        # Chuyển về trang đầu tiên ngay khi khởi tạo
        self.button_instrucst.click()

    # Một số xử lý logic khi chuyển trang
    def change_page(self, index):
        if(index == 1):
            self.stackedWidget.setCurrentWidget(self.page_instruct)
            
            if(self.mode_cam == 'update_frame'):
                self.timer.timeout.disconnect(self.update_frame)
            elif(self.mode_cam == 'start_detect'):
                self.timer.timeout.disconnect(self.start_detect)
            self.mode_cam = 'off'

            if(self.mode_cam_run == 'update_frame_run'):
                self.timerr.timeout.disconnect(self.update_frame_run)
            elif(self.mode_cam_run == 'start_predict'):
                self.timerr.timeout.disconnect(self.start_predict)
            self.mode_cam_run = 'off'

            if self.cap != None:
                self.cap.release()
                self.cap = None

        elif(index == 2):
            self.stackedWidget.setCurrentWidget(self.page_get_data)
            
            if(self.mode_cam_run == 'update_frame_run'):
                self.timerr.timeout.disconnect(self.update_frame_run)
            elif(self.mode_cam_run == 'start_predict'):
                self.timerr.timeout.disconnect(self.start_predict)
            self.mode_cam_run = 'off'
        
            if self.cap == None:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Không thể mở camera")   
                         
            self.timer.timeout.connect(self.update_frame)
            self.mode_cam = 'update_frame'

        elif(index == 3):
            self.stackedWidget.setCurrentWidget(self.page_train)
            
            if(self.mode_cam == 'update_frame'):
                    self.timer.timeout.disconnect(self.update_frame)
            elif(self.mode_cam == 'start_detect'):
                self.timer.timeout.disconnect(self.start_detect)
            self.mode_cam = 'off'

            if(self.mode_cam_run == 'update_frame_run'):
                self.timerr.timeout.disconnect(self.update_frame_run)
            elif(self.mode_cam_run == 'start_predict'):
                self.timerr.timeout.disconnect(self.start_predict)
            self.mode_cam_run = 'off'

            if self.cap != None:
                self.cap.release()
                self.cap = None
        elif(index == 4):
            self.stackedWidget.setCurrentWidget(self.page_run)
            with open('model/categories.pkl', 'rb') as f:
                cat = pickle.load(f)
            self.lb = np.array(cat[0]) # cat là mảng 2 chiều vd: [['label']], chuyển về numpy để thao tác tiện luôn
            self.model_cnn = load_model(r'model/model_cnn.h5')


            if(self.mode_cam == 'update_frame'):
                self.timer.timeout.disconnect(self.update_frame)
            elif(self.mode_cam == 'start_detect'):
                self.timer.timeout.disconnect(self.start_detect)
            self.mode_cam = 'off'

            if self.cap == None:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Không thể mở camera")
            self.timerr.timeout.connect(self.update_frame_run)
            self.mode_cam_run = 'update_frame_run'

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = HandleMain(MainWindow= MainWindow)
    MainWindow.show()
    app.exec_()
