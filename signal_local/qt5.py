# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import cv2
import sys
import time
import os
from PIL import Image,ImageFont,ImageDraw
from PyQt5 import QtMultimedia
# from threading import Thread
# import threading
from need import seq2seq_test
from need import covn_LSTM_test
from need import baiduAPI2mp3
from need.jiebafenci import MappingLable

class Ui_CameraPage(object):
    def setupUi(self, CameraPage):
        CameraPage.setObjectName("CameraPage")
        CameraPage.resize(855, 463)
        CameraPage.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint|QtCore.Qt.WindowCloseButtonHint) #禁止最大化
        CameraPage.setFixedSize(CameraPage.width(), CameraPage.height()) #禁止改变窗口大小
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        CameraPage.setWindowIcon(icon)
        self.layoutWidget = QtWidgets.QWidget(CameraPage)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 30, 700, 420))
        self.layoutWidget.setObjectName("layoutWidget")
        self.layoutWidget2 = QtWidgets.QWidget(CameraPage)
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.layoutWidget3 = QtWidgets.QWidget(CameraPage)
        self.hboxLayout = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.hboxLayout.setContentsMargins(0,0,0,0)
        self.layoutWidget3.setLayout(self.hboxLayout)
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(8, 8, 8, 8)
        self.gridLayout.setObjectName("gridLayout")
        self.vboxLayout = QtWidgets.QVBoxLayout(self.layoutWidget2)
        self.vboxLayout.setObjectName('vboxLayout')
        self.layoutWidget2.setLayout(self.vboxLayout)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.textRegion= QtWidgets.QLineEdit('',self.layoutWidget)
        self.textRegion.setReadOnly(True)
        self.textRegion.setStyleSheet("background:transparent;border-width:0;border-style:outset")
        self.textRegion.setAlignment(Qt.AlignCenter)
        self.textRegion.setFont(font)
        self.textRegion.setObjectName('textRegion')
        self.gridLayout.addWidget(self.textRegion,1,0,1,2)
        self.cameraButton = QtWidgets.QPushButton(self.layoutWidget2)
        self.cameraButton.setFont(font)
        self.cameraButton.setObjectName("cameraButton")
        self.vboxLayout.addWidget(self.cameraButton)
        self.recordButton = QtWidgets.QPushButton(self.layoutWidget2)
        self.recordButton.setFont(font)
        self.recordButton.setObjectName("recordButton")
        self.vboxLayout.addWidget(self.recordButton)
        self.openButton = QtWidgets.QPushButton(self.layoutWidget2)
        self.openButton.setFont(font)
        self.openButton.setObjectName("openButton")
        self.vboxLayout.addWidget(self.openButton)
        # self.loadButton = QtWidgets.QPushButton(self.layoutWidget2)
        # self.loadButton.setFont(font)
        # self.loadButton.setObjectName("loadButton")
        # self.vboxLayout.addWidget(self.loadButton)
        self.testButton = QtWidgets.QPushButton(self.layoutWidget3)
        self.testButton.setFont(font)
        self.testButton.setObjectName("testButton")
        self.hboxLayout.addWidget(self.testButton)
        font2 = QtGui.QFont()
        font2.setPointSize(8)
        self.testButton2 = QtWidgets.QPushButton(self.layoutWidget3)
        self.testButton2.setFont(font2)
        self.testButton2.setObjectName("testButton2")
        self.testButton2.setMinimumWidth(25)  #修改最小宽度限制
        self.testButton2.setMinimumHeight(27)
        self.hboxLayout.addWidget(self.testButton2)
        self.hboxLayout.setStretchFactor(self.testButton,10)
        self.hboxLayout.setStretchFactor(self.testButton2,1)
        self.vboxLayout.addWidget(self.layoutWidget3)

        self.searchLabel = QtWidgets.QLineEdit(self.layoutWidget2)
        self.searchLabel.setPlaceholderText('手语翻译')
        self.searchLabel.setFont(font)
        self.searchLabel.setAlignment(Qt.AlignCenter)
        self.vboxLayout.addWidget(self.searchLabel)

        self.gridLayout.addWidget(self.layoutWidget2,3,0,1,2)

        self.hboxLayout2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.btn1=QtWidgets.QRadioButton('Net1')
        self.btn2=QtWidgets.QRadioButton('Net2')
        self.btn1.setChecked(True)
        self.hboxLayout2.addWidget(self.btn1)
        self.hboxLayout2.addWidget(self.btn2)
        self.hboxLayout2.setContentsMargins(0,0,0,0)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.FieldRole, self.hboxLayout2)
        self.gridLayout.addLayout(self.formLayout,6,0,1,1)

        self.layoutWidget4 = QtWidgets.QWidget(self.layoutWidget)
        self.playButton = QtWidgets.QPushButton(self.layoutWidget)
        self.playButton.setMaximumSize(QtCore.QSize(32, 32))
        self.playButton.setStyleSheet("background:transparent;border-width:0;border-style:outset")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("./bbbb.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.playButton.setIcon(icon2)
        self.playButton.setIconSize(QtCore.QSize(32, 32))
        self.hboxLayout3 = QtWidgets.QHBoxLayout(self.layoutWidget4)
        self.hboxLayout3.addWidget(self.playButton)
        self.hboxLayout3.setContentsMargins(9, 2, 9, 9)
        self.gridLayout.addWidget(self.layoutWidget4, 6, 1, 1, 1)

        self.scrollArea = QtWidgets.QScrollArea(self.layoutWidget)
        self.scrollArea.setMinimumSize(QtCore.QSize(600, 400))
        self.scrollArea.setFont(font)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollArea.setStyleSheet("background:transparent;border-width:0;border-style:outset")
        self.scrollArea.setVisible(False)
        self.gridLayout.addWidget(self.scrollArea, 0, 3, 7, 2)

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout2.setContentsMargins(0, 0, 0, 0)
        self.scrollAreaWidgetContents.setLayout(self.gridLayout2)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.cameraLabel = QtWidgets.QLabel(self.layoutWidget)
        self.cameraLabel.setMinimumSize(QtCore.QSize(600, 400))
        self.cameraLabel.setFont(font)
        self.cameraLabel.setObjectName("cameraLabel")
        self.gridLayout.addWidget(self.cameraLabel, 0, 3, 7, 2)


        self.retranslateUi(CameraPage)
        QtCore.QMetaObject.connectSlotsByName(CameraPage)

    def retranslateUi(self, CameraPage):
        _translate = QtCore.QCoreApplication.translate
        CameraPage.setWindowTitle(_translate("CameraPage", "手语识别系统"))
        self.cameraButton.setText(_translate("CameraPage", "打开摄像头"))
        self.recordButton.setText(_translate("CameraPage", "开始录制"))
        self.openButton.setText(_translate("CameraPage", "打开视频文件"))
        # self.loadButton.setText(_translate("CameraPage", "加载网络模型"))
        self.testButton.setText(_translate("CameraPage", "载入数据"))
        self.testButton2.setText(_translate("CameraPage", "..."))



class CameraPageWindow(QWidget,Ui_CameraPage):
#     returnSignal = pyqtSignal()
    def __init__(self,parent=None):
        super(CameraPageWindow, self).__init__(parent)
        self.timer_camera = QTimer() #初始化定时器
        self.timer_net = QTimer() #网络定时器
        self.timer_net.start(3000) #3秒后加载默认网络
        self.cap = cv2.VideoCapture() #初始化摄像头
        self.cap.set(5,30) #fps为30
        self.CAM_NUM = 0
        self.setupUi(self)
        self.initUI()
        self.slot_init()
        self.flag=0
        self.pat1=0
        self.dirname=''
        self.videodir=''
        self.status=0
        self.net1=None
        self.net2=None
        self.nets=['slr_seq2seq_epoch050.pth','slr_convlstm_epoch068.pth']
        self.player=None
        self.api=baiduAPI2mp3.ApiAuido()
        self.words=''
        self.faudio=''
        self.pic_root='Pictures_of_Signs'
        self.mapLabel=None

    def initUI(self):
        self.setLayout(self.gridLayout)

    def slot_init(self):
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_net.timeout.connect(self.loadNet)
        #信号和槽连接
#         self.returnButton.clicked.connect(self.returnSignal)
        self.cameraButton.clicked.connect(self.slotCameraButton)
        self.recordButton.clicked.connect(self.slotRecordButton)
        self.openButton.clicked.connect(self.openfile)
        # self.loadButton.clicked.connect(self.loadNet)
        self.testButton.clicked.connect(self.loadData1)
        self.testButton2.clicked.connect(self.loadData2)
        self.btn1.toggled.connect(self.loadNet)
        self.playButton.clicked.connect(self.playAudio)
        self.searchLabel.returnPressed.connect(self.showPic)


    def showPic(self):
        if self.timer_camera.isActive():
            QMessageBox.information(None, 'Alert', '请先关闭摄像头或者等待视频播放结束!')
            return
        for i in range(self.gridLayout2.count()):
            self.gridLayout2.itemAt(i).widget().deleteLater()
        self.cameraLabel.setVisible(False)
        self.scrollArea.setVisible(True)
        if not self.mapLabel:
            self.textRegion.setText('分词模型首次载入中...')
            self.textRegion.repaint()
            self.mapLabel = MappingLable(image_dir=self.pic_root)
            self.textRegion.setText('')
            self.textRegion.repaint()
        #         nsize=26
        #         pics=[os.path.join(pic_root,i) for i in os.listdir(pic_root)[0:nsize]]
        #         words=[ ''.join([ chr(random.randint(97,122)) for i in range(1,random.randint(2,7)) ]) for j in range(nsize) ]
        font = QtGui.QFont()
        font.setPointSize(12)
        self.scrollAreaWidgetContents.setMinimumSize(QtCore.QSize(0, 0))
        self.scrollAreaWidgetContents.setMaximumSize(QtCore.QSize(0, 0))
        self.scrollAreaWidgetContents.setMaximumSize(QtCore.QSize(9999999, 9999999))
        for n, i in enumerate(self.mapLabel.chinese_to_img(self.searchLabel.text().strip())):
            line = n // 3 * 2
            col = n % 3
            fh = 30
            if line == 0:
                width = 194 + col * 194
            self.scrollAreaWidgetContents.setMinimumSize(QtCore.QSize(width, 150 + fh + line * (150 + fh) // 2))
            qLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
            qLabel.setMinimumSize(QtCore.QSize(180, 150))
            self.gridLayout2.addWidget(qLabel, line, col, 1, 1)
            pixmap = QPixmap(i[1])
            qLabel.setPixmap(pixmap)
            qText = QtWidgets.QLabel(i[0], self.scrollAreaWidgetContents)
            qText.setFont(font)
            qText.setMaximumSize(QtCore.QSize(180, fh))
            qText.setAlignment(Qt.AlignCenter)
            self.gridLayout2.addWidget(qText, line + 1, col, 1, 1)

    def change_Pixmap(self):
        if self.player and self.player.state() == QtMultimedia.QMediaPlayer.PlayingState:
            icon_name='./aaaa.png'
        else:
            self.player.stop()
            icon_name='./bbbb.png'
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(icon_name), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.playButton.setIcon(icon2)
        self.playButton.setIconSize(QtCore.QSize(32, 32))

    # def muti_Thread(self):
    #     if self.thread and self.thread.is_alive():
    #         self.player.stop()
    #         self.thread.flag.clear()
    #         self.change_Pixmap()
    #     else:
    #         self.thread = Thread(target=self.playAudio, daemon=True)
    #         self.thread.flag=threading.Event()
    #         self.thread.flag.set()
    #         self.thread.start()

    def playAudio(self):
        if self.player and self.player.state() == QtMultimedia.QMediaPlayer.PlayingState:
            return
        else:
            try:
                assert self.net.out_word
            except:
                return
            if self.words!=self.net.out_word:
                self.words=self.net.out_word
                audio_name = self.api.get_auido(self.net.out_word)
                self.faudio=audio_name
            else:
                audio_name=self.faudio
            file = QUrl.fromLocalFile(audio_name)  # 音频文件路径
            content = QtMultimedia.QMediaContent(file)
            self.player = QtMultimedia.QMediaPlayer()
            self.player.setMedia(content)
            self.player.stateChanged.connect(self.change_Pixmap)
            self.player.setVolume(50.0)
            # icon2 = QtGui.QIcon()
            # icon2.addPixmap(QtGui.QPixmap("./aaaa.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            # self.playButton.setIcon(icon2)
            # self.playButton.setIconSize(QtCore.QSize(32, 32))
            self.player.play()
            # while self.thread.flag.isSet():
            #     print(self.player.state())
            #     time.sleep(2)

    def openfile(self):
        if self.timer_camera.isActive():
            if self.flag:
                self.flag=0
                self.recordButton.setText('开始录制')
            self.closeCamera()
        openfile_name = QFileDialog.getOpenFileNames(self,'选择文件','','Video Files (*.mp4 *.avi);;All Files (*)')
        if openfile_name[0]:
            self.videodir=openfile_name[0][0]
            flag = self.cap.open(self.videodir)
            self.dirname='video/VideoCapture_img_'+time.strftime('%Y-%m-%d_%H-%M-%S')
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname)
            self.flag=2
            self.pat1=0
            self.cameraLabel.setVisible(True)
            self.scrollArea.setVisible(False)
            self.timer_camera.start(30)

    def show_camera(self):
        flag,self.image = self.cap.read()
        try:
            assert flag
        except:
            self.timer_camera.stop()
            self.flag=0
            self.showPixmap()
            return
        show = cv2.resize(self.image,(600,400))
        if self.flag:
            show=cv2.putText(show,str(self.pat1//30),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 255), 2)
            cv2.imwrite(self.dirname+'/{:06d}.jpg'.format(self.pat1),self.image)
            self.pat1+=1
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
        self.cameraLabel.setPixmap(QPixmap.fromImage(showImage))
        
    #打开关闭摄像头控制
    def slotCameraButton(self):
        if self.timer_camera.isActive() == False:
            #打开摄像头并显示图像信息
            self.openCamera()
        else:
            #关闭摄像头并清空显示信息
            self.closeCamera()
                        
    def slotRecordButton(self):
        if self.timer_camera.isActive() == False:
            QMessageBox.information(None,'Alert','请先打开摄像头!')
        elif not self.flag:
            self.dirname='video/VideoCapture_img_'+time.strftime('%Y-%m-%d_%H-%M-%S')
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname)
            self.flag=1
            self.pat1=0
            self.recordButton.setText('结束录制')
        elif self.flag==1:
            self.flag=0
            self.recordButton.setText('开始录制')
            self.closeCamera()
            self.showPixmap()

    #打开摄像头
    def openCamera(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QMessageBox.warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
            buttons=QMessageBox.Ok,
            defaultButton=QMessageBox.Ok)
        else:
            self.cameraLabel.setVisible(True)
            self.scrollArea.setVisible(False)
            self.timer_camera.start(1)
            self.cameraButton.setText('关闭摄像头')

    #关闭摄像头
    def closeCamera(self):
        if self.flag:
            QMessageBox.information(None,'Alert','关闭摄像头前请先结束录制!')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.cameraLabel.clear()
            self.cameraButton.setText('打开摄像头')
        
    def closeEvent(self,event):
        self.cap.release()
        self.cameraLabel.clear()
        event.accept()
    
    def showPixmap(self): #显示最近视频第2帧
        image=cv2.imread(os.path.join(self.dirname,'000001.jpg'))
        show = cv2.resize(image,(600,400))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
        self.cameraLabel.setPixmap(QPixmap.fromImage(showImage))

    def showPixmap2(self):  # 显示最近视频第2帧
        image = cv2.imread(os.path.join(self.dirname, '000001.jpg'))
        show = cv2.resize(image, (600, 400))
        if (isinstance(show, np.ndarray)):  # 判断是否OpenCV图片类型
            show = Image.fromarray(cv2.cvtColor(show, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(show)
        font = ImageFont.truetype(r"msyh.ttc", 26)
        text=self.net.out_word
        w, h = font.getsize(text)
        draw.text(((600 - w) // 2, 350), text, (255, 0, 0), font=font)
        show = np.asarray(show)
        #         show=cv2.putText(show,text,(200,350),cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 255), 2)
        #         show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.cameraLabel.setPixmap(QPixmap.fromImage(showImage))

    def loadNet(self):
        if self.timer_net.isActive():
            self.timer_net.stop()
       # if self.timer_camera.isActive() == False and self.status==0:
        if self.btn1.isChecked():
            if os.path.exists(self.nets[0]):
                if self.status != 1 and self.status !=3:
                    self.textRegion.setText('模型%s载入中...' %self.btn1.text())
                    self.textRegion.repaint()
                    self.net1=seq2seq_test.SignLanguagePredict(pthfile=self.nets[0])
                    self.status=3 if self.status else 1
                self.textRegion.setText('模型%s载入完成' %self.btn1.text())
                self.textRegion.repaint()
            else:
                QMessageBox.information(None, 'Alert', '模型文件不存在!' %self.nets[0])
        else:
            if os.path.exists(self.nets[1]):
                if self.status !=2 and self.status!=3:
                    self.textRegion.setText('模型%s载入中...' %self.btn2.text())
                    self.textRegion.repaint()
                    self.net2=covn_LSTM_test.SignLanguage_Isolate_Predict(pthfile=self.nets[1])
                    self.status=3 if self.status else 2
                self.textRegion.setText('模型%s载入完成' %self.btn2.text())
                self.textRegion.repaint()
            else:
                QMessageBox.information(None, 'Alert', '模型文件%s不存在!' %self.nets[1])

    def inputNet(self):
        sample_duration=48 if self.btn1.isChecked() else 16
        self.net=self.net1 if self.btn1.isChecked() else self.net2
        if len(list(filter(lambda f:f.endswith(".jpg"),os.listdir(self.dirname))))>=sample_duration:
            self.textRegion.setText(os.path.split(self.dirname)[-1])
            self.textRegion.repaint()
            self.net.read_images(self.dirname)
            self.net.test()
            self.cameraLabel.setVisible(True)
            self.scrollArea.setVisible(False)
            self.showPixmap2()
        else:
            self.textRegion.setText('载入数据路径错误')
            self.textRegion.repaint()

    def opendata(self):
        openfile_name = QFileDialog.getExistingDirectory(self,'选择文件夹','video/')
        if openfile_name:
            return openfile_name
            
    def loadData1(self):
        if self.timer_camera.isActive() == False and self.status:
            if self.dirname == "":
                QMessageBox.information(None,'Alert','当前载入数据路径为空!')
            else:
                self.inputNet()
        elif self.status==0:
            QMessageBox.information(None,'Alert','请先加载模型!')
        else:
            QMessageBox.information(None,'Alert','请先关闭摄像头或者等待视频播放结束!')
                
    def loadData2(self):
        self.cameraLabel.clear()
        if self.timer_camera.isActive() == False and self.status:
            self.dirname=self.opendata()
            if self.dirname:
                self.inputNet()
        elif self.status==0:
            QMessageBox.information(None,'Alert','请先加载模型!')
        else:
            QMessageBox.information(None,'Alert','请先关闭摄像头或者等待视频播放结束!')
  
if __name__=='__main__':
    app = QApplication(sys.argv)
    myWin = CameraPageWindow()
    myWin.show()
    sys.exit(app.exec_())
