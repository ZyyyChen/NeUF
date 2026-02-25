# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'volumewindow.ui'
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
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLayout, QLineEdit,
    QMainWindow, QPushButton, QSizePolicy, QSpacerItem,
    QWidget)

class Ui_VolumeWindow(object):
    def setupUi(self, VolumeWindow):
        if not VolumeWindow.objectName():
            VolumeWindow.setObjectName(u"VolumeWindow")
        VolumeWindow.resize(307, 262)
        self.mainlayout = QWidget(VolumeWindow)
        self.mainlayout.setObjectName(u"mainlayout")
        self.gridLayout_2 = QGridLayout(self.mainlayout)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setSizeConstraint(QLayout.SetNoConstraint)
        self.label_3 = QLabel(self.mainlayout)
        self.label_3.setObjectName(u"label_3")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.label_3.setFont(font)

        self.gridLayout_2.addWidget(self.label_3, 5, 0, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(25, 5, 5, 5)
        self.label_4 = QLabel(self.mainlayout)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_3.addWidget(self.label_4)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)

        self.doubleSpinBox = QDoubleSpinBox(self.mainlayout)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox")
        self.doubleSpinBox.setDecimals(2)
        self.doubleSpinBox.setMinimum(0.100000000000000)
        self.doubleSpinBox.setMaximum(5.000000000000000)
        self.doubleSpinBox.setSingleStep(0.050000000000000)
        self.doubleSpinBox.setValue(1.000000000000000)

        self.horizontalLayout_3.addWidget(self.doubleSpinBox)

        self.label_5 = QLabel(self.mainlayout)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_3.addWidget(self.label_5)


        self.gridLayout_2.addLayout(self.horizontalLayout_3, 7, 0, 1, 1)

        self.label_2 = QLabel(self.mainlayout)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_2.addWidget(self.label_2, 2, 0, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(-1, -1, -1, 0)
        self.exportButton = QPushButton(self.mainlayout)
        self.exportButton.setObjectName(u"exportButton")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exportButton.sizePolicy().hasHeightForWidth())
        self.exportButton.setSizePolicy(sizePolicy)

        self.horizontalLayout_4.addWidget(self.exportButton)


        self.gridLayout_2.addLayout(self.horizontalLayout_4, 12, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(5, 5, 5, 5)
        self.ckptButton = QPushButton(self.mainlayout)
        self.ckptButton.setObjectName(u"ckptButton")
        icon = QIcon(QIcon.fromTheme(u"dialog-question"))
        self.ckptButton.setIcon(icon)

        self.horizontalLayout.addWidget(self.ckptButton)

        self.lineEdit = QLineEdit(self.mainlayout)
        self.lineEdit.setObjectName(u"lineEdit")

        self.horizontalLayout.addWidget(self.lineEdit)


        self.gridLayout_2.addLayout(self.horizontalLayout, 1, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 9, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.lineEdit_2 = QLineEdit(self.mainlayout)
        self.lineEdit_2.setObjectName(u"lineEdit_2")

        self.horizontalLayout_2.addWidget(self.lineEdit_2)


        self.gridLayout_2.addLayout(self.horizontalLayout_2, 3, 0, 1, 1)

        self.line = QFrame(self.mainlayout)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout_2.addWidget(self.line, 4, 0, 1, 1)

        self.label = QLabel(self.mainlayout)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)

        self.line_2 = QFrame(self.mainlayout)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.gridLayout_2.addWidget(self.line_2, 8, 0, 1, 1)

        VolumeWindow.setCentralWidget(self.mainlayout)

        self.retranslateUi(VolumeWindow)

        QMetaObject.connectSlotsByName(VolumeWindow)
    # setupUi

    def retranslateUi(self, VolumeWindow):
        VolumeWindow.setWindowTitle(QCoreApplication.translate("VolumeWindow", u"Volume Export", None))
        self.label_3.setText(QCoreApplication.translate("VolumeWindow", u"Parameters", None))
        self.label_4.setText(QCoreApplication.translate("VolumeWindow", u"Discretization step:", None))
        self.label_5.setText(QCoreApplication.translate("VolumeWindow", u"mm", None))
        self.label_2.setText(QCoreApplication.translate("VolumeWindow", u"Destination name:", None))
        self.exportButton.setText(QCoreApplication.translate("VolumeWindow", u"Export", None))
        self.ckptButton.setText(QCoreApplication.translate("VolumeWindow", u"File", None))
        self.label.setText(QCoreApplication.translate("VolumeWindow", u"Checkpoint source:", None))
    # retranslateUi

