# This Python file uses the following encoding: utf-8
import sys
import os
import shutil
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from ui_volumewindow import Ui_VolumeWindow

# add the parent directory to the sys.path :
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import volume_export


class VolumeWindow(QMainWindow):
    def __init__(self, parent=None):
        super(VolumeWindow, self).__init__()
        self.ui = Ui_VolumeWindow()
        self.ui.setupUi(self)

    def exportClicked(self):
        if not os.path.exists(self.ui.lineEdit.text()):
            QMessageBox.critical(self,"Unknown file","Source file:\n" + self.ui.lineEdit.text() + "\nDoes not exist")
            return

        dest = self.ui.lineEdit_2.text()
        destinationFolder = "../volume/Generated/"+dest

        if os.path.exists(destinationFolder) :
            button = QMessageBox.question(self,"Existing destination","Export folder:\n" + self.ui.lineEdit_2.text() + "\nAlready exist, overwrite ?")
            if button == QMessageBox.Yes :
                shutil.rmtree(destinationFolder)
            else :
                return

        os.mkdir(destinationFolder)

        volume_export.export_volume(self.ui.lineEdit.text(),dest,destinationFolder,self.ui.doubleSpinBox.value())

        QMessageBox.information(self,"Export Done","Succesfully exported in \n" + dest)


    def ckptClicked(self):
        file = QFileDialog.getOpenFileName(self, "Choose checkpoint", "../logs", "NeUF Checkpoints (*.pkl)")[0]
        if file:
            self.ui.lineEdit.setText(file)


if __name__ == "__main__":
    app = QApplication([])
    window = VolumeWindow()

    # connect
    window.ui.exportButton.clicked.connect(window.exportClicked)
    window.ui.ckptButton.clicked.connect(window.ckptClicked)

    window.show()
    sys.exit(app.exec())
