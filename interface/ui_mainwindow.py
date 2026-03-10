# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QGridLayout,
    QLabel, QLayout, QMainWindow, QProgressBar,
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(709, 546)
        self.mainlayout = QWidget(MainWindow)
        self.mainlayout.setObjectName(u"mainlayout")
        self.gridLayout_2 = QGridLayout(self.mainlayout)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setSizeConstraint(QLayout.SetNoConstraint)
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.gt_checkBox = QCheckBox(self.mainlayout)
        self.gt_checkBox.setObjectName(u"gt_checkBox")
        self.gt_checkBox.setEnabled(False)

        self.verticalLayout_2.addWidget(self.gt_checkBox)

        self.legend_checkBox = QCheckBox(self.mainlayout)
        self.legend_checkBox.setObjectName(u"legend_checkBox")
        self.legend_checkBox.setLayoutDirection(Qt.LeftToRight)
        self.legend_checkBox.setAutoFillBackground(False)
        self.legend_checkBox.setChecked(True)

        self.verticalLayout_2.addWidget(self.legend_checkBox)


        self.gridLayout_2.addLayout(self.verticalLayout_2, 3, 1, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_3, 4, 2, 1, 1)

        self.infos = QLabel(self.mainlayout)
        self.infos.setObjectName(u"infos")
        self.infos.setMinimumSize(QSize(150, 0))
        self.infos.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.gridLayout_2.addWidget(self.infos, 0, 1, 1, 1)

        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(-1, -1, -1, 0)
        self.chartB1 = QWidget(self.mainlayout)
        self.chartB1.setObjectName(u"chartB1")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.chartB1.sizePolicy().hasHeightForWidth())
        self.chartB1.setSizePolicy(sizePolicy)

        self.gridLayout_3.addWidget(self.chartB1, 1, 0, 1, 1)

        self.chartA1 = QWidget(self.mainlayout)
        self.chartA1.setObjectName(u"chartA1")
        sizePolicy.setHeightForWidth(self.chartA1.sizePolicy().hasHeightForWidth())
        self.chartA1.setSizePolicy(sizePolicy)

        self.gridLayout_3.addWidget(self.chartA1, 0, 0, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout_3, 4, 0, 1, 1)

        self.line_3 = QFrame(self.mainlayout)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.gridLayout_2.addWidget(self.line_3, 1, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 0, 2, 1, 1)

        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setSpacing(2)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.imA2 = QLabel(self.mainlayout)
        self.imA2.setObjectName(u"imA2")
        sizePolicy.setHeightForWidth(self.imA2.sizePolicy().hasHeightForWidth())
        self.imA2.setSizePolicy(sizePolicy)
        self.imA2.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.imA2.setMargin(0)

        self.gridLayout_4.addWidget(self.imA2, 0, 2, 1, 1)

        self.imB1 = QLabel(self.mainlayout)
        self.imB1.setObjectName(u"imB1")
        sizePolicy.setHeightForWidth(self.imB1.sizePolicy().hasHeightForWidth())
        self.imB1.setSizePolicy(sizePolicy)
        self.imB1.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.imB1.setMargin(0)

        self.gridLayout_4.addWidget(self.imB1, 0, 4, 1, 1)

        self.imA1 = QLabel(self.mainlayout)
        self.imA1.setObjectName(u"imA1")
        sizePolicy.setHeightForWidth(self.imA1.sizePolicy().hasHeightForWidth())
        self.imA1.setSizePolicy(sizePolicy)
        self.imA1.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.imA1.setMargin(0)

        self.gridLayout_4.addWidget(self.imA1, 0, 0, 1, 1)

        self.imB2 = QLabel(self.mainlayout)
        self.imB2.setObjectName(u"imB2")
        sizePolicy.setHeightForWidth(self.imB2.sizePolicy().hasHeightForWidth())
        self.imB2.setSizePolicy(sizePolicy)
        self.imB2.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.imB2.setMargin(0)

        self.gridLayout_4.addWidget(self.imB2, 0, 5, 1, 1)

        self.imC1 = QLabel(self.mainlayout)
        self.imC1.setObjectName(u"imC1")
        sizePolicy.setHeightForWidth(self.imC1.sizePolicy().hasHeightForWidth())
        self.imC1.setSizePolicy(sizePolicy)
        self.imC1.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.imC1.setMargin(0)

        self.gridLayout_4.addWidget(self.imC1, 1, 0, 1, 1)

        self.line = QFrame(self.mainlayout)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout_4.addWidget(self.line, 0, 3, 1, 1)

        self.imC2 = QLabel(self.mainlayout)
        self.imC2.setObjectName(u"imC2")
        sizePolicy.setHeightForWidth(self.imC2.sizePolicy().hasHeightForWidth())
        self.imC2.setSizePolicy(sizePolicy)
        self.imC2.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.imC2.setMargin(0)

        self.gridLayout_4.addWidget(self.imC2, 1, 2, 1, 1)

        self.imD1 = QLabel(self.mainlayout)
        self.imD1.setObjectName(u"imD1")
        sizePolicy.setHeightForWidth(self.imD1.sizePolicy().hasHeightForWidth())
        self.imD1.setSizePolicy(sizePolicy)
        self.imD1.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.imD1.setMargin(0)

        self.gridLayout_4.addWidget(self.imD1, 1, 4, 1, 1)

        self.imD2 = QLabel(self.mainlayout)
        self.imD2.setObjectName(u"imD2")
        sizePolicy.setHeightForWidth(self.imD2.sizePolicy().hasHeightForWidth())
        self.imD2.setSizePolicy(sizePolicy)
        self.imD2.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.imD2.setMargin(0)

        self.gridLayout_4.addWidget(self.imD2, 1, 5, 1, 1)

        self.line_2 = QFrame(self.mainlayout)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.gridLayout_4.addWidget(self.line_2, 1, 3, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout_4, 0, 0, 1, 1)

        self.progressBar = QProgressBar(self.mainlayout)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setEnabled(True)
        self.progressBar.setValue(24)

        self.gridLayout_2.addWidget(self.progressBar, 3, 0, 1, 1)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.chartA2 = QWidget(self.mainlayout)
        self.chartA2.setObjectName(u"chartA2")
        self.chartA2.setMaximumSize(QSize(16777215, 16777215))

        self.verticalLayout.addWidget(self.chartA2)

        self.chartB2 = QWidget(self.mainlayout)
        self.chartB2.setObjectName(u"chartB2")

        self.verticalLayout.addWidget(self.chartB2)


        self.gridLayout_2.addLayout(self.verticalLayout, 4, 1, 1, 1)

        self.gridLayout_2.setColumnStretch(0, 3)
        self.gridLayout_2.setColumnStretch(1, 1)
        MainWindow.setCentralWidget(self.mainlayout)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Ultrasound Nerf", None))
        self.gt_checkBox.setText(QCoreApplication.translate("MainWindow", u"Show Ground Truth", None))
        self.legend_checkBox.setText(QCoreApplication.translate("MainWindow", u"Show legend", None))
        self.infos.setText(QCoreApplication.translate("MainWindow", u"Informations:", None))
        self.imA2.setText("")
        self.imB1.setText("")
        self.imA1.setText("")
        self.imB2.setText("")
        self.imC1.setText("")
        self.imC2.setText("")
        self.imD1.setText("")
        self.imD2.setText("")
    # retranslateUi

