from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from handle_page_get_data import HandlePageGetData
from handle_page_train import HandelPageTrain
import cv2

class HandleMain(HandlePageGetData, HandelPageTrain):
    def __init__(self, MainWindow):
        HandelPageTrain.__init__(self, MainWindow)
        HandlePageGetData.__init__(self, MainWindow)

        self.button_instrucst.clicked.connect(lambda: self.change_page(1))
        self.button_get_faces.clicked.connect(lambda: self.change_page(2))
        self.button_preprocess_training.clicked.connect(lambda: self.change_page(3))
        self.button_run_trial.clicked.connect(lambda: self.change_page(4))
        self.button_instrucst.click()
    def change_page(self, index):
        if(index == 1):
            self.stackedWidget.setCurrentWidget(self.page_instruct)
            
            if self.cap != None:
                if(self.mode_cam == 1):
                    self.timer.timeout.disconnect(self.start_detect)
                elif(self.mode_cam == 2):
                    self.timer.timeout.disconnect(self.update_frame)
                
                self.cap.release()
                self.cap = None
                self.mode_cam = 0

        elif(index == 2):
            self.stackedWidget.setCurrentWidget(self.page_get_data)
            
            if self.cap == None:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Không thể mở camera")
                    return
            self.timer.timeout.connect(self.update_frame)
            self.mode_cam = 1
        elif(index == 3):
            self.stackedWidget.setCurrentWidget(self.page_train)
            
            if self.cap != None:
                if(self.mode_cam == 1):
                    self.timer.timeout.disconnect(self.start_detect)
                elif(self.mode_cam == 2):
                    self.timer.timeout.disconnect(self.update_frame)
                
                self.cap.release()
                self.cap = None
                self.mode_cam = 0
                
        elif(index == 4):
            self.stackedWidget.setCurrentWidget(self.page_run)
            
            if self.cap == None:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Không thể mở camera")
                    return
                    
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = HandleMain(MainWindow= MainWindow)
    MainWindow.show()
    app.exec_()
