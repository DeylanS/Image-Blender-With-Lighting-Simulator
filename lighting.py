import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox, QFrame,
    QSizePolicy, QSlider, QColorDialog, QLineEdit
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QIntValidator
from PyQt6.QtCore import Qt, QRect

class ScalableImageLabel(QLabel):
    """ A QLabel that automatically scales its pixmap and can draw a light indicator. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pixmap = QPixmap()
        self.angle_slider = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def set_angle_slider(self, slider):
        """Provide a reference to the angle slider to draw the indicator line."""
        self.angle_slider = slider
        # When the slider moves, trigger a repaint of this label to update the line
        if self.angle_slider:
            self.angle_slider.valueChanged.connect(self.update)

    def pixmap(self):
        return self._pixmap

    def _get_target_rect(self):
        """Helper to calculate the on-screen rectangle of the displayed pixmap."""
        if self._pixmap.isNull(): return QRect()
        label_size = self.size()
        scaled_pixmap = self._pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        x = (label_size.width() - scaled_pixmap.width()) / 2
        y = (label_size.height() - scaled_pixmap.height()) / 2
        return QRect(int(x), int(y), scaled_pixmap.width(), scaled_pixmap.height())

    def paintEvent(self, event):
        # First, draw the image
        super().paintEvent(event)
        
        target_rect = self._get_target_rect()
        painter = QPainter(self)
        if not self._pixmap.isNull():
             painter.drawPixmap(target_rect, self._pixmap)

        # --- NEW: Draw the light direction indicator line ---
        if self.angle_slider and not self._pixmap.isNull():
            angle = self.angle_slider.value()
            center = target_rect.center()
            line_length = min(target_rect.width(), target_rect.height()) * 0.4
            
            # Angle is adjusted by 180 to show where light is *coming from*
            rad = np.deg2rad(angle - 180) 
            end_x = center.x() + line_length * np.cos(rad)
            end_y = center.y() + line_length * np.sin(rad)
            
            pen = QPen(QColor(255, 255, 0, 200), 2, Qt.PenStyle.DashLine)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(pen)
            painter.drawLine(center.x(), center.y(), int(end_x), int(end_y))
            painter.drawEllipse(center, 3, 3)

class LightingSimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Lighting Simulator"); self.setGeometry(100, 100, 1200, 800)
        self.original_img = None
        self.processed_img = None
        self.light_color = (255, 255, 220)
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        self.image_label = ScalableImageLabel("Load an image to begin")
        self.image_label.setStyleSheet("QLabel { color: grey; }")
        main_layout.addWidget(self.image_label, 7)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_widget.setFixedWidth(350)
        main_layout.addWidget(controls_widget, 3)

        file_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_save = QPushButton("Save Image")
        file_layout.addWidget(self.btn_load)
        file_layout.addWidget(self.btn_save)

        self.btn_color = QPushButton("Choose Light Color")
        self.blend_combo = QComboBox()
        self.blend_combo.addItems(["Overlay", "Soft Light", "Screen (Lighten)", "Multiply (Darken)"])

        # --- NEW: Create sliders with line edits for direct input ---
        self.angle_slider, self.intensity_slider, self.smoothness_slider = self._create_slider_group()
        
        controls_layout.addLayout(file_layout)
        controls_layout.addWidget(self.btn_color)
        controls_layout.addWidget(self._create_control_widget(QLabel("Blend Mode:"), self.blend_combo))
        controls_layout.addWidget(self.angle_slider)
        controls_layout.addWidget(self.intensity_slider)
        controls_layout.addWidget(self.smoothness_slider)
        controls_layout.addStretch()

        # Pass angle slider reference to the image label so it can draw the indicator
        self.image_label.set_angle_slider(self.angle_slider.findChild(QSlider))

        self.btn_load.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_color.clicked.connect(self.choose_color)
        self.blend_combo.currentIndexChanged.connect(self.update_lighting)
        for slider_widget in [self.angle_slider, self.intensity_slider, self.smoothness_slider]:
            slider_widget.findChild(QSlider).valueChanged.connect(self.update_lighting)
    
    def _create_slider_group(self):
        """Helper to create a group of sliders with consistent settings."""
        angle = self._create_slider_with_lineedit("Angle", 0, 360, 45, "°")
        intensity = self._create_slider_with_lineedit("Intensity", 0, 100, 75, "%")
        smoothness = self._create_slider_with_lineedit("Smoothness", 10, 300, 100, "%")
        return angle, intensity, smoothness
        
    def _create_slider_with_lineedit(self, name, min_val, max_val, default_val, unit):
        """Helper to create a labeled slider with a synced line edit widget."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)
        
        label = QLabel(f"{name}:")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        
        line_edit = QLineEdit(str(default_val))
        line_edit.setValidator(QIntValidator(min_val, max_val, self)) # Ensure only valid numbers are entered
        line_edit.setFixedWidth(40)

        # --- NEW: Two-way synchronization logic ---
        # 1. When slider moves, update the line edit
        slider.valueChanged.connect(lambda v: line_edit.setText(str(v)))
        
        # 2. When line edit is changed (and user presses Enter or clicks away), update the slider
        def update_slider():
            slider.setValue(int(line_edit.text()))
        line_edit.editingFinished.connect(update_slider)

        slider.setValue(default_val) # Set initial value

        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(line_edit)
        return widget

    def _create_control_widget(self, label, widget):
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.addWidget(label)
        layout.addWidget(widget)
        return w

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.original_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            self.update_lighting()

    def save_image(self):
        if self.processed_img is None:
            QMessageBox.warning(self, "No Image", "There is no processed image to save."); return
        output_dir = "lit_images"; os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.join(output_dir, f"lit_{timestamp}.png"), "PNG Images (*.png)")
        if file_path:
            cv2.imwrite(file_path, self.processed_img)
            QMessageBox.information(self, "Success", f"Image saved to:\n{file_path}")

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.light_color = (color.blue(), color.green(), color.red())
            self.update_lighting()
            self.btn_color.setStyleSheet(f"background-color: {color.name()}; color: {'black' if color.lightness() > 127 else 'white'};")

    def _create_gradient(self, w, h, angle, color, intensity, smoothness):
        rad = np.deg2rad(angle)
        vx, vy = np.cos(rad), np.sin(rad)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        projection = x * vx + y * vy
        min_proj, max_proj = projection.min(), projection.max()
        gradient_norm = (projection - min_proj) / (max_proj - min_proj)
        exponent = 0.2 + (smoothness - 10) / (300 - 10) * (4.0 - 0.2)
        gradient_smooth = gradient_norm ** exponent
        alpha_channel = (gradient_smooth * (intensity / 100.0) * 255).astype(np.uint8)
        color_layer = np.full((h, w, 3), color, dtype=np.uint8)
        return cv2.merge([color_layer[:,:,0], color_layer[:,:,1], color_layer[:,:,2], alpha_channel])

    def _blend(self, fg, bg, mode):
        fg_float, bg_float = fg.astype(float), bg.astype(float)
        if mode == "Multiply (Darken)": return ((fg_float * bg_float) / 255).astype(np.uint8)
        if mode == "Screen (Lighten)": return (255 - (((255 - fg_float) * (255 - bg_float)) / 255)).astype(np.uint8)
        if mode == "Overlay": return np.where(bg_float <= 128, (2*fg_float*bg_float)/255, 255-2*(255-fg_float)*(255-bg_float)/255).astype(np.uint8)
        if mode == "Soft Light": return ((1 - 2 * (fg_float/255)) * (bg_float/255)**2 + 2 * (fg_float/255) * (bg_float/255)) * 255

    def update_lighting(self):
        if self.original_img is None: return

        if len(self.original_img.shape) == 3 and self.original_img.shape[2] == 4:
            has_alpha = True
            original_alpha, base_image_bgr = self.original_img[:, :, 3], self.original_img[:, :, :3]
        else:
            has_alpha = False
            base_image_bgr = cv2.cvtColor(self.original_img, cv2.COLOR_GRAY2BGR) if len(self.original_img.shape) < 3 else self.original_img

        angle = self.angle_slider.findChild(QSlider).value()
        intensity = self.intensity_slider.findChild(QSlider).value()
        smoothness = self.smoothness_slider.findChild(QSlider).value()
        blend_mode = self.blend_combo.currentText()

        h, w = base_image_bgr.shape[:2]
        
        gradient_bgra = self._create_gradient(w, h, angle, self.light_color, intensity, smoothness)
        gradient_bgr, gradient_alpha = gradient_bgra[:,:,:3], gradient_bgra[:,:,3]

        blended_base = self._blend(gradient_bgr, base_image_bgr, blend_mode)
        alpha_norm = cv2.cvtColor(gradient_alpha.astype(np.float32) / 255, cv2.COLOR_GRAY2BGR)
        processed_bgr = (base_image_bgr * (1 - alpha_norm) + blended_base * alpha_norm).astype(np.uint8)

        if has_alpha:
            self.processed_img = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2BGRA)
            self.processed_img[:, :, 3] = original_alpha
        else:
            self.processed_img = processed_bgr
        
        self.display_image(self.processed_img)

    def display_image(self, img_cv):
        if img_cv is None: return
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
            h, w, ch = img_cv.shape
            qt_img = QImage(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA).data, w, h, ch * w, QImage.Format.Format_RGBA8888)
        else:
            h, w, ch = img_cv.shape
            qt_img = QImage(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LightingSimulatorApp()
    ex.show()
    sys.exit(app.exec())