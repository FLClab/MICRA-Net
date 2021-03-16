
import numpy
import sys
import time
import random
import torch
import os
import glob
import json
import h5py

from skimage import io
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QHBoxLayout, QGridLayout,
                            QSlider, QRadioButton, QGroupBox, QPushButton, QSizePolicy,
                            QMenuBar, QAction, QFileDialog, QMessageBox, QProgressBar,
                            QVBoxLayout, QListWidget, QComboBox, QInputDialog,
                            QCheckBox, QLineEdit, QDockWidget, QMainWindow,
                            QToolBar, QStatusBar)
from PyQt5.QtGui import (QPixmap, QImage, QColor, QBrush, QDoubleValidator, QPalette,
                        QPainter)
from PyQt5 import QtCore, QtGui

import viewer
import network

class MainWindow(QMainWindow):
    """
    `QMainWindow` of the application
    """
    model_loaded_signal = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__(parent=None)

        self.model = None
        self.loaded_image = None
        self.localmaps = None
        self.directory = os.path.expanduser("~")

        self._worker = Worker(self)
        self._worker.started.connect(self.worker_started_callback)
        self._worker.finished.connect(self.worker_stopped_callback)

        # self.setCentralWidget(QWidget())
        self.initUI()

    def initUI(self):
        """
        Instantiates the user interface
        """
        self.setGeometry(0, 0, 1024, 720)

        bar = QMenuBar()
        bar.setMaximumHeight(25)
        files = bar.addMenu("File")
        load = QAction("Load model", self)
        files.addAction(load)
        save = QAction("Load image", self)
        files.addAction(save)
        quit = QAction("Quit", self)
        quit.setShortcut("Ctrl+q")
        files.addAction(quit)
        files.triggered[QAction].connect(self.file_menu_callback)
        self.setMenuBar(bar)

        # grid = QGridLayout()
        self.grid_viewer = viewer.GridViewer(parent=self)

        statusbar = QStatusBar()
        self.setStatusBar(statusbar)

        self.setCentralWidget(self.grid_viewer)

    def file_menu_callback(self, event):
        """
        Callback of the menu bar update event `triggered`. The `triggered` event
        is handle based on its text

        :param event: A `trigerred` event

        :event 'Live Dialogs': Opens the select live dialogs
        """
        if event.text() == "Quit":
            self.close()
        elif event.text() == "Load model":
            filename = QFileDialog().getOpenFileName(self, "Load model", self.directory)[0]
            if filename:
                self.directory = os.path.dirname(filename)
                if "PVivaxModelZoo" in filename:
                    with h5py.File(filename, "r") as file:
                        networks = {}
                        for key, values in file["-".join(("MICRANet", "finetuned"))].items():
                            networks[key] = [{k : torch.tensor(v[()]) for k, v in fold_values.items()} for fold, fold_values in values.items()]
                        self.trainer_params = json.loads(file["-".join(("MICRANet", "finetuned"))].attrs["trainer_params"])
                    net_params = networks[key][0]
                elif "EMModelZoo" in filename:
                    with h5py.File(filename, "r") as file:
                        networks = {}
                        for key, values in file["-".join(("MICRANet", "1:5"))].items():
                            networks[key] = {k : torch.tensor(v[()]) for k, v in values.items()}
                        self.trainer_params = json.loads(file["-".join(("MICRANet", "1:5"))].attrs["trainer_params"])
                    net_params = networks[key]
                else:
                    with h5py.File(filename, "r") as file:
                        networks = {}
                        for key, values in file["MICRANet"].items():
                            networks[key] = {k : torch.tensor(v[()]) for k, v in values.items()}
                        self.trainer_params = json.loads(file["MICRANet"].attrs["trainer_params"])
                    net_params = networks[key]
                if "model_params" in self.trainer_params:
                    self.trainer_params.update(self.trainer_params["model_params"])

                self.model = network.MICRANet(grad=True, **self.trainer_params)
                self.model.load_state_dict(net_params)
                self.model.eval()

                # Updates the combobox
                combobox = self.grid_viewer.class_group.findChild(QComboBox)
                combobox.clear()
                for i in range(self.trainer_params["num_classes"]):
                    combobox.addItem(str(i))

                self.model_loaded_signal.emit()

        elif event.text() == "Load image":
            filename = QFileDialog().getOpenFileName(self, "Load image", self.directory)[0]
            if filename:
                self.directory = os.path.dirname(filename)
                self.loaded_image = io.imread(filename)
                # io reads 3 channel image in the form of [H, W, C]
                if self.loaded_image.ndim == 3:
                    self.loaded_image = numpy.transpose(self.loaded_image, axes=(2, 0, 1))
                if self.loaded_image.dtype == "uint8":
                    self.loaded_image  = (self.loaded_image / 255.).astype(numpy.float32)
                if self.loaded_image.ndim == 2:
                    self.loaded_image = numpy.expand_dims(self.loaded_image, axis=0)
                self.grid_viewer.loaded_viewer.update_image(self.loaded_image)
        else:
            pass

    def worker_started_callback(self):
        """
        Handles the worker started callback.
        """
        self.statusBar().showMessage("Predicting ...")

    def worker_stopped_callback(self):
        """
        Handles the worker stopped callback.
        """
        self.statusBar().showMessage("Predicting done!", 2000)

    def predict_callback(self):
        """
        Implements the `predict` callback
        """
        if (not isinstance(self.model, type(None))) and (not isinstance(self.loaded_image, type(None))):
            self._worker.start()

    def update_callback(self):
        """
        Implements the `update` callback
        """
        if not isinstance(self.localmaps, type(None)):
            combobox = self.grid_viewer.class_group.findChild(QComboBox)
            for localmap, image_selector in zip(self.localmaps[0, int(combobox.currentText())], self.grid_viewer.image_selectors):
                image_selector.update_image(localmap)
            self.grid_viewer.generated_viewer.update_callback()

    def paintEvent(self, event):
        """
        Overrides the `paintEvent` of `QMainWindow`. We update the background image
        of the MainWindow

        :param event: A paint event
        """
        pass

    def closeEvent(self, event):
        """
        Overrides the `closeEvent` of the `QMainWindow`. We handle the QThread
        """
        self._worker.stop()

class Worker(QtCore.QThread):
    """
    Creates a Worker Thread to safely predict the images using the loaded model
    """
    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self._stopped = True
        self._mutex = QtCore.QMutex()

    def stop(self):
        self._mutex.lock()
        self._stopped = True
        self._mutex.unlock()

    def run(self):
        self._stopped = False
        parent = self.parent()

        # Extracts the rect coords
        rect_coords = parent.grid_viewer.loaded_viewer.get_rect_coords()

        size = parent.trainer_params["size"]
        cropped = parent.loaded_image[:, rect_coords[1]:rect_coords[1] + size, rect_coords[0]:rect_coords[0] + size]

        X = torch.tensor([cropped])
        if X.ndim == 3:
            X = X.unsqueeze(1)
        parent.localmaps, pred = network.class_activation_map(parent.model, X, cuda=False, size=size, num_classes=parent.trainer_params["num_classes"])
        parent.localmaps[~pred] = 0

        combobox = parent.grid_viewer.class_group.findChildren(QComboBox)[0]
        for localmap, image_selector in zip(parent.localmaps[0, int(combobox.currentText())], parent.grid_viewer.image_selectors):
            image_selector.update_image(localmap)

if __name__ == "__main__":


    app = QApplication(sys.argv)
    window = MainWindow()

    app.setStyle("Fusion")
    # Now use a palette to switch to dark colors:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QPalette.Text, QtCore.Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

    window.show()
    sys.exit(app.exec_())
