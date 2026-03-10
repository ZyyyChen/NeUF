# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'startwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QSpinBox, QWidget)

class Ui_StartWindow(object):
    def setupUi(self, StartWindow):
        if not StartWindow.objectName():
            StartWindow.setObjectName(u"StartWindow")
        StartWindow.resize(574, 243)
        self.centralwidget = QWidget(StartWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.spinBox_miter = QSpinBox(self.centralwidget)
        self.spinBox_miter.setObjectName(u"spinBox_miter")
        self.spinBox_miter.setMaximum(100000000)
        self.spinBox_miter.setSingleStep(10)
        self.spinBox_miter.setValue(10000)

        self.gridLayout.addWidget(self.spinBox_miter, 7, 3, 1, 1)

        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 8, 1, 1, 1)

        self.comboBox_2 = QComboBox(self.centralwidget)
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.setObjectName(u"comboBox_2")

        self.gridLayout.addWidget(self.comboBox_2, 8, 3, 1, 1)

        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 6, 1, 1, 1)

        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 7, 1, 1, 1)

        self.lineEdit_data = QLineEdit(self.centralwidget)
        self.lineEdit_data.setObjectName(u"lineEdit_data")

        self.gridLayout.addWidget(self.lineEdit_data, 2, 1, 1, 4)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 10, 1, 1, 1)

        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")

        self.gridLayout.addWidget(self.pushButton, 10, 3, 1, 5)

        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 5, 1, 1, 1)

        self.pushButton_data = QPushButton(self.centralwidget)
        self.pushButton_data.setObjectName(u"pushButton_data")

        self.gridLayout.addWidget(self.pushButton_data, 2, 6, 1, 1)

        self.comboBox = QComboBox(self.centralwidget)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")

        self.gridLayout.addWidget(self.comboBox, 5, 3, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 9, 3, 1, 1)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)

        self.spinBox_save = QSpinBox(self.centralwidget)
        self.spinBox_save.setObjectName(u"spinBox_save")
        self.spinBox_save.setMinimum(1)
        self.spinBox_save.setMaximum(100000)
        self.spinBox_save.setValue(100)

        self.gridLayout.addWidget(self.spinBox_save, 7, 8, 1, 1)

        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 7, 7, 1, 1)

        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 7, 4, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 10, 8, 1, 1)

        self.spinBox_piter = QSpinBox(self.centralwidget)
        self.spinBox_piter.setObjectName(u"spinBox_piter")
        self.spinBox_piter.setEnabled(True)
        self.spinBox_piter.setMinimum(1)
        self.spinBox_piter.setMaximum(1000000)
        self.spinBox_piter.setValue(50000)

        self.gridLayout.addWidget(self.spinBox_piter, 6, 3, 1, 1)

        self.lineEdit_ckpt = QLineEdit(self.centralwidget)
        self.lineEdit_ckpt.setObjectName(u"lineEdit_ckpt")
        self.lineEdit_ckpt.setEnabled(False)

        self.gridLayout.addWidget(self.lineEdit_ckpt, 4, 1, 1, 4)

        self.spinBox_plot = QSpinBox(self.centralwidget)
        self.spinBox_plot.setObjectName(u"spinBox_plot")
        self.spinBox_plot.setMinimum(1)
        self.spinBox_plot.setMaximum(1000000)
        self.spinBox_plot.setValue(100)

        self.gridLayout.addWidget(self.spinBox_plot, 7, 6, 1, 1)

        self.checkBox_ckpt = QCheckBox(self.centralwidget)
        self.checkBox_ckpt.setObjectName(u"checkBox_ckpt")

        self.gridLayout.addWidget(self.checkBox_ckpt, 4, 7, 1, 1)

        self.pushButton_ckpt = QPushButton(self.centralwidget)
        self.pushButton_ckpt.setObjectName(u"pushButton_ckpt")
        self.pushButton_ckpt.setEnabled(False)

        self.gridLayout.addWidget(self.pushButton_ckpt, 4, 6, 1, 1)

        self.label_8 = QLabel(self.centralwidget)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout.addWidget(self.label_8, 8, 4, 1, 1)

        self.lineEdit_seed = QLineEdit(self.centralwidget)
        self.lineEdit_seed.setObjectName(u"lineEdit_seed")

        self.gridLayout.addWidget(self.lineEdit_seed, 8, 6, 1, 1)

        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout.addWidget(self.label_9, 3, 1, 1, 1)

        StartWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(StartWindow)

        QMetaObject.connectSlotsByName(StartWindow)
    # setupUi

    def retranslateUi(self, StartWindow):
        StartWindow.setWindowTitle(QCoreApplication.translate("StartWindow", u"NeUF starter", None))
        self.label_7.setText(QCoreApplication.translate("StartWindow", u"Encoding:", None))
        self.comboBox_2.setItemText(0, QCoreApplication.translate("StartWindow", u"None", None))
        self.comboBox_2.setItemText(1, QCoreApplication.translate("StartWindow", u"Hash", None))
        self.comboBox_2.setItemText(2, QCoreApplication.translate("StartWindow", u"Frequency", None))

        self.label_6.setText(QCoreApplication.translate("StartWindow", u"Points per iterations:", None))
        self.label_2.setText(QCoreApplication.translate("StartWindow", u"Max iterations:", None))
        self.pushButton.setText(QCoreApplication.translate("StartWindow", u"Go", None))
        self.label_5.setText(QCoreApplication.translate("StartWindow", u"Training mode:", None))
        self.pushButton_data.setText(QCoreApplication.translate("StartWindow", u"Select", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("StartWindow", u"Slice", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("StartWindow", u"Random", None))

        self.label.setText(QCoreApplication.translate("StartWindow", u"Dataset:", None))
        self.label_4.setText(QCoreApplication.translate("StartWindow", u"Save frequency: ", None))
        self.label_3.setText(QCoreApplication.translate("StartWindow", u"Plot frequency: ", None))
        self.checkBox_ckpt.setText(QCoreApplication.translate("StartWindow", u"Checkpoint", None))
        self.pushButton_ckpt.setText(QCoreApplication.translate("StartWindow", u"Select", None))
        self.label_8.setText(QCoreApplication.translate("StartWindow", u"Seed:", None))
        self.lineEdit_seed.setText(QCoreApplication.translate("StartWindow", u"19981708", None))
        self.label_9.setText(QCoreApplication.translate("StartWindow", u"Checkpoint:", None))
    # retranslateUi

