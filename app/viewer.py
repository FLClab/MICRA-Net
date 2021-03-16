
import numpy
import qimage2ndarray
import functools

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QHBoxLayout, QGridLayout,
                            QSlider, QRadioButton, QGroupBox, QPushButton, QSizePolicy,
                            QMenuBar, QAction, QFileDialog, QMessageBox, QProgressBar,
                            QVBoxLayout, QListWidget, QComboBox, QInputDialog,
                            QCheckBox, QLineEdit, QDockWidget, QMainWindow,
                            QToolBar, QButtonGroup)
from PyQt5.QtGui import (QPixmap, QImage, QColor, QBrush, QDoubleValidator, QPalette,
                        QPainter, QPen)
from PyQt5 import QtCore, QtGui

import utils

class GridViewer(QWidget):
    """
    A `QWidget` responsible of the `CentralWidget` of the `QMainWindow`
    """
    def __init__(self, parent=None):
        super(GridViewer, self).__init__(parent=parent)

        self.combiner = utils.Combiner()
        self.thresholder = utils.Thresholder()

        self.initUI()

    def initUI(self):
        """
        Instantiates the user interface
        """

        parent = self.parent()

        self.setGeometry(0, 0, 512, 512)

        grid = QGridLayout()

        # grad-cam layers
        self.image_selectors = []
        for i in range(8):
            image_selector = ImageSelector(parent=parent)
            grid.addWidget(image_selector, int(i // 2) + 1, i % 2)
            self.image_selectors.append(image_selector)

        self.logo_pixmap = QPixmap()
        loaded = self.logo_pixmap.load("./img/logo_white.svg.png")
        self.display_label = QLabel("")
        self.display_label.setPixmap(self.logo_pixmap)
        self.display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.display_label.setAlignment(QtCore.Qt.AlignCenter)
        grid.addWidget(self.display_label, 0, 0)

        # loaded image
        self.loaded_viewer = LoadedViewer(parent=parent)
        grid.addWidget(self.loaded_viewer, 0, 1)

        # generated layer
        self.generated_viewer = ImageViewer(parent=parent)
        grid.addWidget(self.generated_viewer, 1, 2, 4, 1)

        # Control panel
        control_layout = QHBoxLayout()

        # Control panel class selector
        self.class_group = QGroupBox("Parameters")
        class_layout = QVBoxLayout()
        combobox = QComboBox()
        combobox.addItem("0")
        combobox.currentIndexChanged.connect(self.combobox_callback)
        class_layout.addWidget(QLabel("Class selector"))
        class_layout.addWidget(combobox)
        self.class_group.setLayout(class_layout)

        # Control panel Combiner
        self.combiner_group = QGroupBox("Combiner")
        combiner_layout = QVBoxLayout()
        self.combiner_buttons = []
        for i, name in enumerate(["Sum", "Prod", "PCA"]):
            button = QRadioButton(name)
            button.setChecked(i == 0)
            button.clicked.connect(self.combiner_button_clicked_callback)
            self.combiner_buttons.append(button)
            combiner_layout.addWidget(button)
        self.combiner_group.setLayout(combiner_layout)

        # Control panel thresholder
        self.thresholder_group = QGroupBox("Threshold")
        thresholder_layout = QGridLayout()
        self.thresholder_buttons = QButtonGroup()
        self.thresholder_buttons.setExclusive(False)
        for i, name in enumerate(["Otsu", "Triangle", "Percentile"]):
            button = QCheckBox(name)
            button.setChecked(False)
            self.thresholder_buttons.addButton(button)
            thresholder_layout.addWidget(button, 0, i)
        self.current_button_state = [b.isChecked() for b in self.thresholder_buttons.buttons()]
        self.thresholder_buttons.buttonPressed.connect(self.button_pressed_callback)
        self.thresholder_buttons.buttonClicked.connect(self.button_clicked_callback)
        slider = QSlider(QtCore.Qt.Horizontal)
        slider.valueChanged.connect(self.slider_callback)
        thresholder_layout.addWidget(slider, 1, 1, 1, len(self.thresholder_buttons.buttons()) - 1)
        thresholder_layout.addWidget(QLabel(str(slider.value())), 1, 0, 1, 1)
        self.thresholder_group.setLayout(thresholder_layout)

        # Control panel Updater
        self.updater_group = QGroupBox("Updater")
        updater_layout = QVBoxLayout()
        button = QPushButton("Predict")
        button.setObjectName("Predict")
        button.setShortcut("P")
        button.pressed.connect(parent.predict_callback)
        updater_layout.addWidget(button)
        button = QPushButton("Update")
        button.setObjectName("Update")
        button.pressed.connect(parent.update_callback)
        button.setShortcut("U")
        updater_layout.addWidget(button)
        self.updater_group.setLayout(updater_layout)

        control_layout.addWidget(self.class_group)
        control_layout.addWidget(self.combiner_group)
        control_layout.addWidget(self.thresholder_group)
        control_layout.addWidget(self.updater_group)
        grid.addLayout(control_layout, 0, 2)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 3)

        self.setLayout(grid)

    def slider_callback(self, val):
        """
        Handles the slider callback

        :param val: An `int` of the value of the slider
        """
        label = self.thresholder_group.findChild(QLabel)
        label.setText(str(val))

        # Calls update
        self.updater_group.findChild(QPushButton, "Update").click()

    def button_pressed_callback(self, button):
        """
        Handles the button pressed callback

        :param button: A `QCheckBox` button
        """
        self.current_button_state = [b.isChecked() for b in self.thresholder_buttons.buttons()]

    def button_clicked_callback(self, button):
        """
        Handles the button clicked callback

        :param button: A `QCheckBox` button
        """
        for current, _button in zip(self.current_button_state, self.thresholder_buttons.buttons()):
            if _button == button:
                _button.setChecked(not current)
            else:
                _button.setChecked(False)

        # Calls update
        self.updater_group.findChild(QPushButton, "Update").click()

    def combiner_button_clicked_callback(self, event):
        # Calls update
        self.updater_group.findChild(QPushButton, "Update").click()

    def combobox_callback(self, event):
        # Calls update
        self.updater_group.findChild(QPushButton, "Update").click()

    def paintEvent(self, event):
        """
        Overrides the `paint` event
        """
        h, w = self.display_label.height(), self.display_label.width()
        self.display_label.setPixmap(self.logo_pixmap.scaled(min(h, w), min(h, w), aspectRatioMode=QtCore.Qt.KeepAspectRatio))

class ImageViewer(QWidget):
    """
    Creates an `ImageViewer`
    """
    def __init__(self, parent=None, **kwargs):
        super(ImageViewer, self).__init__(parent=parent)

        self.image_selectors = []
        for key, values in kwargs.items():
            setattr(self, key, values)
        self.image_array = None

        self.initUI()

    def initUI(self):
        """
        Instantiates the user interface
        """

        self.setGeometry(0, 0, 256, 256)

        layout = QVBoxLayout()

        # Image
        self.display_label = QLabel("")
        self.image = QImage(256, 256, QImage.Format_Indexed8)
        self.image.setColorTable([QtGui.qRgb(i, i, i) for i in range(256)])
        self.image.fill(0)
        self.display_label.setPixmap(QPixmap.fromImage(self.image))
        self.display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.display_label.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(self.display_label)

        self.setLayout(layout)

    def update_image(self, arr=None, normalize=True):
        """
        Updates the image from the arr

        :param arr: A `numpy.ndarray` of the image
        """
        if isinstance(arr, numpy.ndarray):
            self.image_array = arr
            if arr.ndim == 3:
                arr = numpy.transpose(arr, axes=(1, 2, 0))
            self.image = qimage2ndarray.array2qimage(arr, normalize=normalize)
            self.image.setColorTable([QtGui.qRgb(i, i, i) for i in range(256)])
        else:
            self.image.fill(0)
        # self.display_label.setPixmap(QPixmap.fromImage(self.image))
        self.update()

    def update_callback(self):
        """
        Handles the push button callback
        """
        parent = self.parent()
        arrays = []
        for image_selector in parent.image_selectors:
            if image_selector.isChecked():
                arrays.append(image_selector.image_array)
        if len(arrays) > 0:
            for child in parent.combiner_group.findChildren(QRadioButton):
                if child.isChecked():
                    method = child.text().lower()
                    break
            # Thresholding
            thresh_method = None
            for button in parent.thresholder_buttons.buttons():
                if button.isChecked():
                    thresh_method = button.text().lower()
            slider = parent.thresholder_group.findChild(QSlider)
            flattened = parent.combiner.flatten(arrays, method)
            thresholded = parent.thresholder.threshold(flattened, thresh_method, slider.value())
            self.update_image(thresholded, normalize=True)
        else:
            self.update_image()

    def paintEvent(self, event):
        """
        Overrides the `paint` event
        """
        h, w = self.display_label.height(), self.display_label.width()
        self.display_label.setPixmap(QPixmap.fromImage(self.image).scaled(min(h, w), min(h, w), aspectRatioMode=QtCore.Qt.KeepAspectRatio))

class ImageSelector(ImageViewer):
    """
    Creates an `ImageSelector`
    """
    def __init__(self, parent=None):
        super(ImageSelector, self).__init__(parent=parent)

    def initUI(self):
        """
        Instantiates the user interface
        """

        self.setGeometry(0, 0, 256, 256)

        layout = QVBoxLayout()

        # Image
        self.display_label = SelectableQLabel("", parent=self.parent())
        self.image = QImage(256, 256, QImage.Format_Indexed8)
        self.image.setColorTable([QtGui.qRgb(0, i, 0) for i in range(256)])
        self.image.fill(0)
        self.display_label.setPixmap(QPixmap.fromImage(self.image))
        self.display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.display_label.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(self.display_label)

        self.setLayout(layout)

    def isChecked(self):
        """
        Implements a `isChecked` method to verify if the `SelectableQLabel` is
        checked

        :returns : A `bool`
        """
        return self.display_label.isChecked()

class SelectableQLabel(QLabel):
    """
    Implements a `SelectableQLabel`. A selectable `QLabel` is a QLabel which
    implements some of the functions of a `QCheckBox`
    """
    def __init__(self, text, parent=None):
        super(SelectableQLabel, self).__init__(text, parent=parent)

        self.checked = False

        self.popup_widget = ImageViewer()
        self.popup_widget.setGeometry(50, 50, 512, 512)

    def setChecked(self, isChecked):
        """
        Implements the `setChecked` similarly to a `QCheckBox`

        :param isChecked: A `bool`
        """
        self.checked = isChecked

    def isChecked(self):
        """
        Implements the `isChecked` similarly to a `QCheckBox`

        :returns : A `bool`
        """
        return self.checked

    def mousePressEvent(self, event):
        """
        Implements a `mousePressEvent`. It changes the boolean value of the checked

        :param event: A `mousePressEvent`
        """
        # Leftclick
        if event.button() == 1:
            self.checked = not self.checked
            # parent = self.parent()
            if self.checked:
                self.setStyleSheet("border: 2px solid #ff9100")
            else:
                self.setStyleSheet("")

        # Rightclick
        elif event.button() == 2:
            parent = self.parent()
            arr = parent.image_array
            self.popup_widget.update_image(arr=arr)
            if not self.popup_widget.isVisible():
                self.popup_widget.show()
            if not self.popup_widget.hasFocus():
                parent = self.parent()
                self.popup_widget.display_label.setPixmap(QPixmap.fromImage(parent.image))
                self.popup_widget.raise_()
                self.popup_widget.activateWindow()

        else:
            print("oops click not handled", event.button())

class LoadedViewer(ImageViewer):
    def __init__(self, parent=None):
        super(LoadedViewer, self).__init__(parent=parent)

        self.parent().model_loaded_signal.connect(self.update_model_params)
        self.display_label.popup_widget.close_signal.connect(self.window_closed_callback)

    def initUI(self):

        layout = QHBoxLayout()

        # Image
        self.display_label = PopupQLabel("", parent=self)
        self.image = QImage(256, 256, QImage.Format_Indexed8)
        self.image.setColorTable([QtGui.qRgb(i, i, i) for i in range(256)])
        self.image.fill(0)
        self.display_label.setPixmap(QPixmap.fromImage(self.image))
        self.display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.display_label.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(self.display_label)

        self.setLayout(layout)

    def update_model_params(self):
        """
        Updates the image size of the `RegionSelector`
        """
        parent = self.parent()
        while not isinstance(parent, QMainWindow):
            parent = parent.parent()
        trainer_params = parent.trainer_params
        self.display_label.popup_widget.set_image_size(trainer_params["size"])
        self.display_label.popup_widget.update()

    def update_image(self, arr=None, normalize=True):
        """
        Updates the image from the arr

        :param arr: A `numpy.ndarray` of the image
        """
        # Updates the popup widget first
        self.display_label.popup_widget.update_image(arr=arr, normalize=normalize)

        if isinstance(arr, numpy.ndarray):
            self.image_array = arr
            if arr.ndim == 3:
                arr = numpy.transpose(arr, axes=(1, 2, 0))
            self.image = qimage2ndarray.array2qimage(arr, normalize=normalize)
            self.image.setColorTable([QtGui.qRgb(i, i, i) for i in range(256)])
        else:
            self.image.fill(0)
        self.display_label.setPixmap(QPixmap.fromImage(self.image))

    def get_rect_coords(self):
        """
        Gets the rect coords from the `RegionSelector` widget
        """
        return self.display_label.popup_widget.rect_coords

    def window_closed_callback(self):
        """
        Handles the `window_close` signal
        """
        self.parent().findChild(QPushButton, "Predict").click()

    def paintEvent(self, event):
        h, w = self.display_label.height(), self.display_label.width()
        pixmap = self.display_label.popup_widget.make_pixmap(self.image, pen_width=5)
        self.display_label.setPixmap(pixmap.scaled(min(h, w), min(h, w), aspectRatioMode=QtCore.Qt.KeepAspectRatio))

class PopupQLabel(QLabel):
    def __init__(self, text, parent=None):
        super(PopupQLabel, self).__init__(text, parent=parent)

        self.initUI()

    def initUI(self):

        self.popup_widget = RegionSelector()
        self.popup_widget.setGeometry(50, 50, 512, 512)

    def mousePressEvent(self, event):
        """
        Reimplements the `mousePressEvent` of `QLabel`

        :param event: A `mousePressEvent`
        """
        parent = self.parent()
        arr = parent.image_array
        self.popup_widget.update_image(arr=arr)

        if not self.popup_widget.isVisible():
            self.popup_widget.show()
        if not self.popup_widget.hasFocus():
            parent = self.parent()
            self.popup_widget.update()
            self.popup_widget.raise_()
            self.popup_widget.activateWindow()

class RegionSelector(ImageViewer):

    region_signal = QtCore.pyqtSignal()
    close_signal = QtCore.pyqtSignal()

    def __init__(self, parent=None):

        self.image_size = 256
        self.rect_coords = (0, 0, 256, 256)

        super(RegionSelector, self).__init__(parent=parent)

    def initUI(self):

        self.setGeometry(0, 0, 256, 256)

        layout = QVBoxLayout()

        # Image
        self.display_label = QLabel("")
        self.image = QImage(self.image_size, self.image_size, QImage.Format_Indexed8)
        self.image.setColorTable([QtGui.qRgb(i, i, i) for i in range(256)])
        self.image.fill(0)

        # Creates the pixmap
        pixmap = self.make_pixmap(self.image)
        self.display_label.setPixmap(pixmap)
        self.display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.display_label.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(self.display_label)

        self.setLayout(layout)

    def set_image_size(self, image_size):
        """
        Sets the image size value

        :param image_size: An `int` of the image size
        """
        self.image_size = image_size
        self.rect_coords = (self.rect_coords[0], self.rect_coords[1], image_size, image_size)

    def make_pixmap(self, image, pen_width=2):

        pixmap = QPixmap.fromImage(image)

        pen = QPen(QColor("#ff9100"))
        pen.setWidth(pen_width)

        painterInstance = QPainter()
        painterInstance.begin(pixmap)
        painterInstance.setPen(pen)
        painterInstance.drawRect(*self.rect_coords)
        painterInstance.end()

        return pixmap

    def update_rect(self, event):
        contents_rect = self.contentsRect()
        event_pos = event.globalPos()
        event_pos = self.mapFromGlobal(event_pos)
        pixmap = self.display_label.pixmap()

        xoffset = int((contents_rect.width() - pixmap.rect().width()) / 2)
        yoffset = int((contents_rect.height() - pixmap.rect().height()) / 2)

        x = event_pos.x() - xoffset
        y = event_pos.y() - yoffset

        if (x > 0) and (x < pixmap.width()) and (y > 0) and (y < pixmap.height()):
            self.interp_x = int(numpy.interp(x, [0, pixmap.width()], [0, self.image.width()]))
            self.interp_y = int(numpy.interp(y, [0, pixmap.height()], [0, self.image.height()]))

        half_size = self.image_size // 2
        self.rect_coords = (
            numpy.clip(self.interp_x, half_size, self.image.width() - half_size) - half_size,
            numpy.clip(self.interp_y, half_size, self.image.height() - half_size) - half_size,
            self.image_size,
            self.image_size
        )

    def mouseMoveEvent(self, event):
        self.update_rect(event)
        self.update()
        self.region_signal.emit()

    def mousePressEvent(self, event):
        self.update_rect(event)
        self.update()
        self.region_signal.emit()

    def paintEvent(self, event):
        h, w = self.display_label.height(), self.display_label.width()
        pixmap = self.make_pixmap(self.image)
        self.display_label.setPixmap(pixmap.scaled(min(h, w), min(h, w), aspectRatioMode=QtCore.Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.close_signal.emit()
