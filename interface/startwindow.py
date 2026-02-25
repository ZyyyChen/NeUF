from   PySide6           import QtCore
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QMainWindow,  QFileDialog, QMessageBox
from PySide6.QtGui import QIntValidator
from PySide6 import QtCharts
import sys
import datetime
import os
from mainwindow import MainWindow

from NeUF import NeUF
from ui_startwindow import Ui_StartWindow

class StartWindow(QMainWindow):
    def __init__(self):
        super(StartWindow, self).__init__()
        self.ui = Ui_StartWindow()
        self.ui.setupUi(self)

        self.ui.label_6.setVisible(False)
        self.ui.spinBox_piter.setVisible(False)

        self.ui.lineEdit_seed.setValidator(QIntValidator())
        self.thread = None
        self.window = None

        # Automation of training
        self.ui.lineEdit_data.setText(os.path.join(os.path.dirname(__file__),"..","datasets/baked/test/latest.pkl"))
        self.ui.spinBox_miter.setValue(50000)
        self.ui.spinBox_plot.setValue(10000)
        self.ui.spinBox_save.setValue(10000)
        self.ui.comboBox.setCurrentIndex(1)
        self.training_changed(1)
        self.ui.spinBox_piter.setValue(1000)


    def select_data_clicked(self):
        file = QFileDialog.getOpenFileName(self, "Choose dataset", "../datasets/baked/test", "NeUF datasets (*.pkl)")[0]
        if file:
            self.ui.lineEdit_data.setText(file)

    def select_ckpt_clicked(self):
        file = QFileDialog.getOpenFileName(self, "Choose checkpoint", "../logs", "NeUF Checkpoints (*.pkl)")[0]
        if file:
            self.ui.lineEdit_ckpt.setText(file)



    def ckpt_changed(self,value):
        self.ui.lineEdit_ckpt.setEnabled(value)
        self.ui.pushButton_ckpt.setEnabled(value)

        self.ui.lineEdit_data.setEnabled(not value)
        self.ui.pushButton_data.setEnabled(not value)
        self.ui.label_7.setEnabled(not value)
        self.ui.comboBox_2.setEnabled(not value)
        self.ui.lineEdit_seed.setEnabled(not value)

    def training_changed(self,value):
        if value == 0 :
            self.ui.label_6.setVisible(False)
            self.ui.spinBox_piter.setVisible(False)
        else :
            self.ui.label_6.setVisible(True)
            self.ui.spinBox_piter.setVisible(True)

    def go_clicked(self):
        if not self.ui.lineEdit_data.text() and not self.ui.checkBox_ckpt.isChecked() :
            QMessageBox.critical(self, "Dataset error", "Dataset must not be empty")
            return

        if not self.ui.lineEdit_ckpt.text() and self.ui.checkBox_ckpt.isChecked() :
            QMessageBox.critical(self, "Checkpoint error", "Checkpoint must not be empty")
            return

        ckpt = self.ui.lineEdit_ckpt.text() if self.ui.checkBox_ckpt.isChecked() else ""
        neuf = NeUF(dataset=self.ui.lineEdit_data.text(),
                    seed=int(self.ui.lineEdit_seed.text()),
                    nb_iters_max=self.ui.spinBox_miter.value(),
                    plot_freq=self.ui.spinBox_plot.value(),
                    save_freq=self.ui.spinBox_save.value(),
                    training_mode=str(self.ui.comboBox.currentText()),
                    points_per_iter=self.ui.spinBox_piter.value(),
                    encoding=str(self.ui.comboBox_2.currentText()),
                    checkpoint=ckpt)

        refs = neuf.getReferences()
        gts = neuf.getGT()

        self.window = MainWindow(neuf.getEncodingName(), datetime.date.today().strftime("%d-%m-%Y"), neuf.getDatasetName(),
                            neuf.logPath, neuf.training_mode,
                            refs[0], refs[1], refs[2], refs[3],
                            gts[0], gts[1], gts[2], gts[3])

        self.thread = QThread()
        neuf.moveToThread(self.thread)

        # connect
        self.thread.started.connect(neuf.run)
        self.window.ui.legend_checkBox.stateChanged.connect(self.window.legendCheckChanged)
        self.window.ui.gt_checkBox.stateChanged.connect(self.window.gtCheckChanged)
        neuf.progress.connect(self.window.progressBar)
        neuf.new_values.connect(self.window.updateInfos)

        self.thread.start()


        self.window.show()
        self.window.init_window_graphics()
        self.hide()


if __name__ == "__main__" :
    app = QApplication(sys.argv)

    window = StartWindow()
    window.show()

    #connect
    window.ui.pushButton_data.clicked.connect(window.select_data_clicked)
    window.ui.pushButton_ckpt.clicked.connect(window.select_ckpt_clicked)
    window.ui.checkBox_ckpt.stateChanged.connect(window.ckpt_changed)
    window.ui.comboBox.currentIndexChanged.connect(window.training_changed)
    window.ui.pushButton.clicked.connect(window.go_clicked)

    window.go_clicked()

    sys.exit(app.exec())
