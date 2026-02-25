import datetime
import sys
import os

# add the parent directory to the sys.path :
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import torch
from   PySide6           import QtCore
from PySide6.QtCore import QThread
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout
from PySide6 import QtCharts
from PySide6.QtCharts import QLineSeries, QChart, QChartView

from NeUF import NeUF
from ui_mainwindow import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self, encode_str, date_str, dataset_str, folder_str, mode,
                 refA2, refB2, refC2, refD2,
                 gtA2=None, gtB2=None, gtC2=None, gtD2=None):
        super(MainWindow, self).__init__()
        self.folder_str = folder_str
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.progressBar.setVisible(False)

        self.setStyleSheet("background-color: white;")

        self.loss_train_series = QLineSeries()
        self.loss_train_series.setName("Training loss")
        self.loss_valid_series = QLineSeries()
        self.loss_valid_series.setName("Validation loss")
        self.loss_gt_series = QLineSeries()
        self.loss_gt_series.setName("Variance")
        self.loss_gt_series_zoom = QLineSeries()
        self.loss_train_series_zoom = QLineSeries()
        self.loss_valid_series_zoom = QLineSeries()


        self.ymax_t = 0
        self.ymax_v = 0
        self.ymax_g = 0

        self.ymax_zt = 0
        self.ymax_zv = 0
        self.ymax_zg = 0
        self.nb_value_zoom = 10

        self.tv_chart = QChart()
        self.gt_chart = QChart()
        self.tv_chart.legend().setAlignment(QtCore.Qt.AlignLeft)
        self.gt_chart.legend().setAlignment(QtCore.Qt.AlignLeft)

        self.tv_chart_zoom = QChart()
        self.gt_chart_zoom = QChart()
        self.tv_chart_zoom.legend().hide()
        self.gt_chart_zoom.legend().hide()

        self.tv_chart.addSeries(self.loss_train_series)
        self.tv_chart.addSeries(self.loss_valid_series)
        self.tv_chart_zoom.addSeries(self.loss_train_series_zoom)
        self.tv_chart_zoom.addSeries(self.loss_valid_series_zoom)
        self.tv_chart.createDefaultAxes()
        self.tv_chart_zoom.createDefaultAxes()
        self.tv_chart.axes(QtCore.Qt.Orientation.Vertical)[0].setLabelFormat("%d")
        self.tv_chart.axes(QtCore.Qt.Orientation.Horizontal)[0].setLabelFormat("%d")
        self.tv_chart.setMargins(QtCore.QMargins(0,0,0,0))
        self.tv_chart_zoom.axes(QtCore.Qt.Orientation.Vertical)[0].setLabelFormat("%d")
        self.tv_chart_zoom.axes(QtCore.Qt.Orientation.Horizontal)[0].setLabelFormat("%d")
        self.tv_chart_zoom.setMargins(QtCore.QMargins(0, 0, 0, 0))

        self.gt_chart.addSeries(self.loss_gt_series)
        self.gt_chart_zoom.addSeries(self.loss_gt_series_zoom)
        self.gt_chart.createDefaultAxes()
        self.gt_chart_zoom.createDefaultAxes()
        self.gt_chart.axes(QtCore.Qt.Orientation.Vertical)[0].setLabelFormat("%d")
        self.gt_chart.axes(QtCore.Qt.Orientation.Horizontal)[0].setLabelFormat("%d")
        self.gt_chart.setMargins(QtCore.QMargins(0, 0, 0, 0))
        self.gt_chart_zoom.axes(QtCore.Qt.Orientation.Vertical)[0].setLabelFormat("%d")
        self.gt_chart_zoom.axes(QtCore.Qt.Orientation.Horizontal)[0].setLabelFormat("%d")
        self.gt_chart_zoom.setMargins(QtCore.QMargins(0, 0, 0, 0))

        self.tv_chart_view = QChartView(self.tv_chart)
        self.ui.chartA1.setContentsMargins(0,0,0,0)
        self.layA1 = QHBoxLayout(self.ui.chartA1)
        self.layA1.setContentsMargins(0,0,0,0)
        self.layA1.addWidget(self.tv_chart_view)
        self.tv_chart_view_zoom = QChartView(self.tv_chart_zoom)
        self.ui.chartA2.setContentsMargins(0,0,0,0)
        self.layA2 = QHBoxLayout(self.ui.chartA2)
        self.layA2.setContentsMargins(0,0,0,0)
        self.layA2.addWidget(self.tv_chart_view_zoom)

        self.gt_chart_view = QChartView(self.gt_chart)
        self.ui.chartB1.setContentsMargins(0,0,0,0)
        self.layB1 = QHBoxLayout(self.ui.chartB1)
        self.layB1.setContentsMargins(0,0,0,0)
        self.layB1.addWidget(self.gt_chart_view)

        self.gt_chart_view_zoom = QChartView(self.gt_chart_zoom)
        self.ui.chartB2.setContentsMargins(0,0,0,0)
        self.layB2 = QHBoxLayout(self.ui.chartB2)
        self.layB2.setContentsMargins(0,0,0,0)
        self.layB2.addWidget(self.gt_chart_view_zoom)

        self.tv_chart_view.setRenderHint(QPainter.Antialiasing)
        self.gt_chart_view.setRenderHint(QPainter.Antialiasing)
        self.tv_chart_view_zoom.setRenderHint(QPainter.Antialiasing)
        self.gt_chart_view_zoom.setRenderHint(QPainter.Antialiasing)

        print(refA2.shape)

        self.imA1 = QPixmap(
            QImage(np.random.randint(0, 255, (refA2.shape[1], refA2.shape[0]), np.uint8).data, refA2.shape[1],
                   refA2.shape[0], refA2.shape[1], QImage.Format.Format_Grayscale8))
        self.imB1 = QPixmap(
            QImage(np.random.randint(0, 255, (refA2.shape[1], refA2.shape[0]), np.uint8).data, refA2.shape[1],
                   refA2.shape[0], refA2.shape[1], QImage.Format.Format_Grayscale8))
        self.imC1 = QPixmap(
            QImage(np.random.randint(0, 255, (refA2.shape[1], refA2.shape[0]), np.uint8).data, refA2.shape[1],
                   refA2.shape[0], refA2.shape[1], QImage.Format.Format_Grayscale8))
        self.imD1 = QPixmap(
            QImage(np.random.randint(0, 255, (refA2.shape[1], refA2.shape[0]), np.uint8).data, refA2.shape[1],
                   refA2.shape[0], refA2.shape[1], QImage.Format.Format_Grayscale8))

        self.refA2 = QPixmap(
            QImage(torch.clamp(refA2.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, refA2.shape[1],
                   refA2.shape[0], refA2.shape[1], QImage.Format.Format_Grayscale8))
        self.refB2 = QPixmap(
            QImage(torch.clamp(refB2.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, refB2.shape[1],
                   refB2.shape[0], refB2.shape[1], QImage.Format.Format_Grayscale8))
        self.refC2 = QPixmap(
            QImage(torch.clamp(refC2.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, refC2.shape[1],
                   refC2.shape[0], refC2.shape[1], QImage.Format.Format_Grayscale8))
        self.refD2 = QPixmap(
            QImage(torch.clamp(refD2.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, refD2.shape[1],
                   refD2.shape[0], refD2.shape[1], QImage.Format.Format_Grayscale8))

        self.gtA2 = None
        self.gtB2 = None
        self.gtC2 = None
        self.gtD2 = None

        if gtA2 != None:
            self.ui.gt_checkBox.setEnabled(True)
            self.gtA2 = QPixmap(
                QImage(torch.clamp(gtA2.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, gtA2.shape[1],
                       gtA2.shape[0], gtA2.shape[1],
                       QImage.Format.Format_Grayscale8))
        if gtB2 != None:
            self.gtB2 = QPixmap(
                QImage(torch.clamp(gtB2.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, gtB2.shape[1],
                       gtB2.shape[0], gtB2.shape[1],
                       QImage.Format.Format_Grayscale8))
        if gtC2 != None:
            self.gtC2 = QPixmap(
                QImage(torch.clamp(gtC2.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, gtC2.shape[1],
                       gtC2.shape[0], gtC2.shape[1],
                       QImage.Format.Format_Grayscale8))
        if gtD2 != None:
            self.gtD2 = QPixmap(
                QImage(torch.clamp(gtD2.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, gtD2.shape[1],
                       gtD2.shape[0], gtD2.shape[1],
                       QImage.Format.Format_Grayscale8))

        self.encode_str = encode_str
        self.date_str = date_str
        self.dataset_str = dataset_str
        self.mode = mode
        self.ui.infos.setText(self.getInfoString(0, "0s", encode_str, date_str, dataset_str,mode))



    def init_window_graphics(self):
        w = min(self.width() // 3, self.ui.imA2.width())
        h = min(self.height() // 3, self.ui.imA2.height())
        self.ui.imA2.setPixmap(self.refA2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imB2.setPixmap(self.refB2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imC2.setPixmap(self.refC2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imD2.setPixmap(self.refD2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))

        self.ui.imA1.setPixmap(self.imA1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imB1.setPixmap(self.imB1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imC1.setPixmap(self.imC1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imD1.setPixmap(self.imD1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))

        self.tv_chart_view.resize(self.ui.chartA1.size())
        self.gt_chart_view.resize(self.ui.chartB1.size())


    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        w = min(self.width()//4,self.ui.imA2.width())
        h = min(self.height()//4,self.ui.imA2.height())
        if self.ui.gt_checkBox.isChecked() :
            self.ui.imA2.setPixmap(self.gtA2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imB2.setPixmap(self.gtB2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imC2.setPixmap(self.gtC2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imD2.setPixmap(self.gtD2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        else:
            self.ui.imA2.setPixmap(self.refA2.scaled(w,h,QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imB2.setPixmap(self.refB2.scaled(w,h,QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imC2.setPixmap(self.refC2.scaled(w,h,QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imD2.setPixmap(self.refD2.scaled(w,h,QtCore.Qt.AspectRatioMode.KeepAspectRatio))

        self.ui.imA1.setPixmap(self.imA1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imB1.setPixmap(self.imB1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imC1.setPixmap(self.imC1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imD1.setPixmap(self.imD1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))



    def progressBar(self,i,min,max):
        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setRange(min,max)
        self.ui.progressBar.setValue(i)

    def updateInfos(self, params):

        self.loss_train_series.append(params["iteration"], params["loss_train"])
        self.ymax_t = max(int(params["loss_train"]),self.ymax_t)

        self.loss_valid_series.append(params["iteration"], params["loss_valid"])
        self.ymax_v = max(int(params["loss_valid"]),self.ymax_v)
        if params["loss_gt"] != None :
            self.loss_gt_series.append(params["iteration"], params["loss_gt"])
            self.ymax_g = max(int(params["loss_gt"]),self.ymax_g)


        tvy = self.tv_chart.axes(QtCore.Qt.Orientation.Vertical)
        tvy[0].setMin(0)
        tvy[0].setMax(max(self.ymax_v,self.ymax_t)+500)

        gy = self.gt_chart.axes(QtCore.Qt.Orientation.Vertical)
        gy[0].setMin(0)
        gy[0].setMax(self.ymax_g+500)

        self.tv_chart.axes(QtCore.Qt.Orientation.Horizontal)[0].setMax(params["iteration"])
        self.gt_chart.axes(QtCore.Qt.Orientation.Horizontal)[0].setMax(params["iteration"])

        ymaxzoom_tv = max(int(params["loss_train"]),int(params["loss_valid"]))
        yminzoom_tv = min(int(params["loss_train"]),int(params["loss_valid"]))
        if params["loss_gt"] != None:
            ymaxzoom_gt = int(params["loss_gt"])
            yminzoom_gt = int(params["loss_gt"])


        if len(self.loss_train_series_zoom.points()) > self.nb_value_zoom:
            self.loss_train_series_zoom.remove(0)
            self.loss_valid_series_zoom.remove(0)
            if params["loss_gt"] != None:
                self.loss_gt_series_zoom.remove(0)
        for i in range(len(self.loss_train_series_zoom.points())):
            yminzoom_tv = min(yminzoom_tv,min(self.loss_valid_series_zoom.points()[i].y(),self.loss_train_series_zoom.points()[i].y()))
            ymaxzoom_tv = max(ymaxzoom_tv,max(self.loss_valid_series_zoom.points()[i].y(),self.loss_train_series_zoom.points()[i].y()))
            if params["loss_gt"] != None:
                yminzoom_gt = min(yminzoom_gt,self.loss_gt_series_zoom.points()[i].y())
                ymaxzoom_gt = max(ymaxzoom_gt,self.loss_gt_series_zoom.points()[i].y())



        self.loss_train_series_zoom.append(params["iteration"], params["loss_train"])
        self.loss_valid_series_zoom.append(params["iteration"], params["loss_valid"])
        if params["loss_gt"] != None:
            self.loss_gt_series_zoom.append(params["iteration"], params["loss_gt"])


        self.tv_chart_zoom.axes(QtCore.Qt.Orientation.Horizontal)[0].setMax(params["iteration"])
        self.gt_chart_zoom.axes(QtCore.Qt.Orientation.Horizontal)[0].setMax(params["iteration"])
        self.tv_chart_zoom.axes(QtCore.Qt.Orientation.Horizontal)[0].setMin(max(0,params["iteration"]-self.nb_value_zoom*params["i_plot"]))
        self.gt_chart_zoom.axes(QtCore.Qt.Orientation.Horizontal)[0].setMin(max(0,params["iteration"]-self.nb_value_zoom*params["i_plot"]))



        self.tv_chart_zoom.axes(QtCore.Qt.Orientation.Vertical)[0].setMin(yminzoom_tv-100)
        self.tv_chart_zoom.axes(QtCore.Qt.Orientation.Vertical)[0].setMax(ymaxzoom_tv+100)
        if params["loss_gt"] != None:
            self.gt_chart_zoom.axes(QtCore.Qt.Orientation.Vertical)[0].setMin(yminzoom_gt-100)
            self.gt_chart_zoom.axes(QtCore.Qt.Orientation.Vertical)[0].setMax(ymaxzoom_gt+100)





        w = min(self.width()//3,self.ui.imA2.width())
        h = min(self.height()//3,self.ui.imA2.height())

        A1 = params["A1"]
        B1 = params["B1"]
        C1 = params["C1"]
        D1 = params["D1"]

        self.imA1 = QPixmap(
            QImage(torch.clamp(A1.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, A1.shape[1],
                   A1.shape[0], A1.shape[1], QImage.Format.Format_Grayscale8))
        self.imB1 = QPixmap(
            QImage(torch.clamp(B1.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, B1.shape[1],
                   B1.shape[0], B1.shape[1], QImage.Format.Format_Grayscale8))
        self.imC1 = QPixmap(
            QImage(torch.clamp(C1.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, C1.shape[1],
                   C1.shape[0], C1.shape[1], QImage.Format.Format_Grayscale8))
        self.imD1 = QPixmap(
            QImage(torch.clamp(D1.detach().cpu(), 0, 255).to(torch.uint8).numpy().data, D1.shape[1],
                   D1.shape[0], D1.shape[1], QImage.Format.Format_Grayscale8))


        self.ui.imA1.setPixmap(self.imA1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imB1.setPixmap(self.imB1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imC1.setPixmap(self.imC1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.imD1.setPixmap(self.imD1.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))

        self.ui.infos.setText(self.getInfoString(params["iteration"],params["time"],self.encode_str,self.date_str,self.dataset_str,self.mode))
        self.grab().save(self.folder_str+"/images/"+str(params["iteration"])+".png")
        self.tv_chart_view.resize(self.ui.chartA1.width()*2,self.ui.chartA1.height()*2)
        self.tv_chart_view.grab().save(self.folder_str+"/losses/"+str(params["iteration"])+".png")
        self.tv_chart_view.resize(self.ui.chartA1.size())


    def legendCheckChanged(self,value):
        if value :
            self.tv_chart.legend().show()
            self.gt_chart.legend().show()

        else :
            self.tv_chart.legend().hide()
            self.gt_chart.legend().hide()

    def gtCheckChanged(self,value):
        w = min(self.width() // 3, self.ui.imA2.width())
        h = min(self.height() // 3, self.ui.imA2.height())
        if value :
            self.ui.imA2.setPixmap(self.gtA2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imB2.setPixmap(self.gtB2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imC2.setPixmap(self.gtC2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imD2.setPixmap(self.gtD2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        else:
            self.ui.imA2.setPixmap(self.refA2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imB2.setPixmap(self.refB2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imC2.setPixmap(self.refC2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.imD2.setPixmap(self.refD2.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    @staticmethod
    def getInfoString(iter,time,encode,date,dataset,mode):
        return "Informations:\n\t-iterations: " + str(iter) +"\n\t-time elapsed: " + time + "\n\t-encoding: " + encode + "\n\t-date: " + date + "\n\t-dataset: " + dataset + "\n\t-mode: " + mode




if __name__ == "__main__":
    app = QApplication(sys.argv)

    # neuf = NeUF(dataset="E:/NeRF-3/datasets/baked/test/dicom.pkl")
    neuf = NeUF(dataset="../datasets/baked/dicomBalayageTotalValidNew.pkl")
    #neuf = NeUF(dataset="../datasets/baked/seinGrandeSphereValidNew.pkl")

    refs = neuf.getReferences()
    gts = neuf.getGT()

    window = MainWindow(neuf.getEncodingName(),datetime.date.today().strftime("%d-%m-%Y"),neuf.getDatasetName(), neuf.logPath, neuf.training_mode,
                        refs[0],refs[1],refs[2],refs[3],
                        gts[0],gts[1],gts[2],gts[3])

    thread = QThread()
    neuf.moveToThread(thread)

    #connect
    thread.started.connect(neuf.run)
    window.ui.legend_checkBox.stateChanged.connect(window.legendCheckChanged)
    window.ui.gt_checkBox.stateChanged.connect(window.gtCheckChanged)
    neuf.progress.connect(window.progressBar)
    neuf.new_values.connect(window.updateInfos)

    thread.start()

    window.show()
    window.init_window_graphics()

    sys.exit(app.exec())
