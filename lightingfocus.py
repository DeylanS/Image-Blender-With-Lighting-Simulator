import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox, QFrame,
    QSizePolicy, QSlider, QColorDialog, QLineEdit, QCheckBox
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QIntValidator
from PyQt6.QtCore import Qt, QRect, pyqtSignal

class ScalableImageLabel(QLabel):
    pointSelected = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pixmap = QPixmap()
        self.angle_slider = None
        self.focus_point = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def set_angle_slider(self, slider):
        self.angle_slider = slider
        if self.angle_slider:
            self.angle_slider.valueChanged.connect(self.update)

    def set_focus_point(self, point):
        self.focus_point = point
        self.update()

    def pixmap(self):
        return self._pixmap

    def _get_target_rect(self):
        if self._pixmap.isNull(): return QRect()
        label_size = self.size()
        scaled_pixmap = self._pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        x = (label_size.width() - scaled_pixmap.width()) / 2
        y = (label_size.height() - scaled_pixmap.height()) / 2
        return QRect(int(x), int(y), scaled_pixmap.width(), scaled_pixmap.height())

    def mousePressEvent(self, event):
        if not self._pixmap.isNull():
            target_rect = self._get_target_rect()
            if target_rect.contains(event.position().toPoint()):
                relative_x = event.position().x() - target_rect.x()
                relative_y = event.position().y() - target_rect.y()
                scale = self._pixmap.width() / target_rect.width()
                img_x = int(relative_x * scale)
                img_y = int(relative_y * scale)
                self.pointSelected.emit(img_x, img_y)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        target_rect = self._get_target_rect()
        painter = QPainter(self)
        if not self._pixmap.isNull():
             painter.drawPixmap(target_rect, self._pixmap)
        
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.angle_slider and not self._pixmap.isNull():
            angle = self.angle_slider.value()
            center = target_rect.center()
            line_length = min(target_rect.width(), target_rect.height()) * 0.4
            rad = np.deg2rad(angle - 180) 
            end_x = center.x() + line_length * np.cos(rad)
            end_y = center.y() + line_length * np.sin(rad)
            pen = QPen(QColor(255, 255, 0, 200), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(center.x(), center.y(), int(end_x), int(end_y))
            painter.drawEllipse(center, 3, 3)

        if self.focus_point and not self._pixmap.isNull():
            scale = target_rect.width() / self._pixmap.width()
            label_x = target_rect.x() + int(self.focus_point[0] * scale)
            label_y = target_rect.y() + int(self.focus_point[1] * scale)
            pen = QPen(QColor(255, 0, 0, 220), 2)
            painter.setPen(pen)
            painter.drawEllipse(label_x - 10, label_y - 10, 20, 20)
            painter.drawLine(label_x - 15, label_y, label_x + 15, label_y)
            painter.drawLine(label_x, label_y - 15, label_x, label_y + 15)

class LightingSimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Lighting & Focus Simulator"); self.setGeometry(100, 100, 1200, 800)
        self.original_img, self.processed_img = None, None
        self.light_color = (255, 255, 220)
        self.focus_point = None
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
        self.btn_load, self.btn_save = QPushButton("Load Image"), QPushButton("Save Image")
        file_layout.addWidget(self.btn_load); file_layout.addWidget(self.btn_save)

        self.btn_color = QPushButton("Choose Light Color")
        self.blend_combo = QComboBox(); self.blend_combo.addItems(["Overlay", "Soft Light", "Screen (Lighten)", "Multiply (Darken)"])
        
        self.angle_slider_w, self.intensity_slider_w, self.smoothness_slider_w = self._create_slider_group("Light", [("Angle", 0, 360, 45, "°"), ("Intensity", 0, 100, 75, "%"), ("Smoothness", 10, 300, 100, "%")])
        
        focus_frame = QFrame(); focus_frame.setFrameShape(QFrame.Shape.StyledPanel)
        focus_layout = QVBoxLayout(focus_frame)
        self.focus_checkbox = QCheckBox("Enable Focus Effect")
        self.focus_size_slider_w, self.blur_amount_slider_w, self.focus_transition_slider_w = self._create_slider_group("Focus", [("Focus Size", 10, 500, 100, "px"), ("Blur Amount", 1, 51, 15, "px"), ("Transition", 1, 101, 51, "px")])

        focus_layout.addWidget(self.focus_checkbox)
        focus_layout.addWidget(self.focus_size_slider_w)
        focus_layout.addWidget(self.blur_amount_slider_w)
        focus_layout.addWidget(self.focus_transition_slider_w)
        
        controls_layout.addLayout(file_layout)
        controls_layout.addWidget(self._create_section_label("Lighting Effect"))
        controls_layout.addWidget(self.btn_color)
        controls_layout.addWidget(self._create_control_widget(QLabel("Blend Mode:"), self.blend_combo))
        controls_layout.addWidget(self.angle_slider_w); controls_layout.addWidget(self.intensity_slider_w); controls_layout.addWidget(self.smoothness_slider_w)
        
        controls_layout.addWidget(self._create_section_label("Focus Effect"))
        controls_layout.addWidget(focus_frame)
        controls_layout.addStretch()

        self.image_label.set_angle_slider(self.angle_slider_w.findChild(QSlider))

        self.btn_load.clicked.connect(self.load_image); self.btn_save.clicked.connect(self.save_image)
        self.btn_color.clicked.connect(self.choose_color); self.blend_combo.currentIndexChanged.connect(self.update_pipeline)
        self.image_label.pointSelected.connect(self.set_focus_point)
        self.focus_checkbox.toggled.connect(self.update_pipeline)
        
        for group in [self.angle_slider_w, self.intensity_slider_w, self.smoothness_slider_w, self.focus_size_slider_w, self.blur_amount_slider_w, self.focus_transition_slider_w]:
            group.findChild(QSlider).valueChanged.connect(self.update_pipeline)
    
    def _create_slider_group(self, name, sliders_config):
        widgets = []
        for conf in sliders_config:
            widgets.append(self._create_slider_with_lineedit(*conf))
        return widgets

    # --- FINAL FIX: THIS IS THE CORRECTED FUNCTION ---
    def _create_slider_with_lineedit(self, name, min_val, max_val, default_val, unit):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)
        
        label = QLabel(f"{name}:")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        
        line_edit = QLineEdit(str(default_val))
        line_edit.setValidator(QIntValidator(min_val, max_val, self))
        line_edit.setFixedWidth(40)

        slider.valueChanged.connect(lambda v, le=line_edit: le.setText(str(v)))
        line_edit.editingFinished.connect(lambda s=slider, le=line_edit: s.setValue(int(le.text())))
        slider.setValue(default_val)

        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(line_edit)
        return widget

    def _create_control_widget(self, label, widget):
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 5, 0, 5); layout.addWidget(label); layout.addWidget(widget)
        return w

    def _create_section_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; margin-top: 10px;"); return label

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.original_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            h, w = self.original_img.shape[:2]
            self.set_focus_point(w // 2, h // 2)
            self.update_pipeline()
    
    def set_focus_point(self, x, y):
        self.focus_point = (x, y)
        self.image_label.set_focus_point(self.focus_point)
        self.update_pipeline()

    def save_image(self):
        if self.processed_img is None: QMessageBox.warning(self, "No Image", "There is no processed image to save."); return
        output_dir = "processed_images"; os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.join(output_dir, f"processed_{timestamp}.png"), "PNG Images (*.png)")
        if file_path: cv2.imwrite(file_path, self.processed_img); QMessageBox.information(self, "Success", f"Image saved to:\n{file_path}")

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.light_color = (color.blue(), color.green(), color.red())
            self.update_pipeline()
            self.btn_color.setStyleSheet(f"background-color: {color.name()}; color: {'black' if color.lightness() > 127 else 'white'};")

    def _create_gradient(self, w, h, angle, color, intensity, smoothness):
        rad = np.deg2rad(angle); vx, vy = np.cos(rad), np.sin(rad)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        projection = x * vx + y * vy
        min_proj, max_proj = projection.min(), projection.max()
        gradient_norm = (projection - min_proj) / (max_proj - min_proj) if max_proj != min_proj else np.zeros_like(projection)
        exponent = 0.2 + (smoothness - 10) / (300 - 10) * (4.0 - 0.2)
        gradient_smooth = gradient_norm ** exponent
        alpha_channel = (gradient_smooth * (intensity / 100.0) * 255).astype(np.uint8)
        color_layer = np.full((h, w, 3), color, dtype=np.uint8)
        return cv2.merge([color_layer[:,:,0], color_layer[:,:,1], color_layer[:,:,2], alpha_channel])

    def _blend(self, fg, bg, mode):
        fg_float, bg_float = fg.astype(float), bg.astype(float)
        if mode == "Multiply (Darken)": result = ((fg_float * bg_float) / 255)
        elif mode == "Screen (Lighten)": result = (255 - (((255 - fg_float) * (255 - bg_float)) / 255))
        elif mode == "Overlay": result = np.where(bg_float <= 128, (2*fg_float*bg_float)/255, 255-2*(255-fg_float)*(255-bg_float)/255)
        elif mode == "Soft Light": result = ((1 - 2 * (fg_float/255)) * (bg_float/255)**2 + 2 * (fg_float/255) * (bg_float/255)) * 255
        else: result = bg_float
        return result.astype(np.uint8)

    def _apply_focus_effect(self, image):
        if not self.focus_checkbox.isChecked() or self.focus_point is None:
            self.image_label.set_focus_point(None if not self.focus_checkbox.isChecked() else self.focus_point)
            return image

        focus_size = self.focus_size_slider_w.findChild(QSlider).value()
        blur_amount = self.blur_amount_slider_w.findChild(QSlider).value() * 2 + 1
        transition = self.focus_transition_slider_w.findChild(QSlider).value() * 2 + 1
        
        if len(image.shape) == 3 and image.shape[2] == 4:
            has_alpha, original_alpha, sharp_bgr = True, image[:, :, 3], image[:, :, :3]
        else:
            has_alpha, sharp_bgr = False, image
        
        blurry_bgr = cv2.GaussianBlur(sharp_bgr, (blur_amount, blur_amount), 0)
        
        h, w = sharp_bgr.shape[:2]
        blur_map = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(blur_map, self.focus_point, focus_size, 255, -1)
        blur_map = cv2.GaussianBlur(blur_map, (transition, transition), 0)
        
        blur_map_float = cv2.cvtColor(blur_map.astype(np.float32) / 255.0, cv2.COLOR_GRAY2BGR)
        focused_bgr = (sharp_bgr * blur_map_float + blurry_bgr * (1 - blur_map_float)).astype(np.uint8)
        
        if has_alpha:
            return cv2.merge([focused_bgr[:,:,0], focused_bgr[:,:,1], focused_bgr[:,:,2], original_alpha])
        else:
            return focused_bgr

    def update_pipeline(self):
        if self.original_img is None: return

        if len(self.original_img.shape) == 3 and self.original_img.shape[2] == 4:
            has_alpha, original_alpha, base_image_bgr = True, self.original_img[:, :, 3], self.original_img[:, :, :3]
        else:
            has_alpha, base_image_bgr = False, cv2.cvtColor(self.original_img, cv2.COLOR_GRAY2BGR) if len(self.original_img.shape) < 3 else self.original_img

        angle, intensity, smoothness = [s.findChild(QSlider).value() for s in [self.angle_slider_w, self.intensity_slider_w, self.smoothness_slider_w]]
        blend_mode = self.blend_combo.currentText()
        h, w = base_image_bgr.shape[:2]
        
        gradient_bgra = self._create_gradient(w, h, angle, self.light_color, intensity, smoothness)
        gradient_bgr, gradient_alpha = gradient_bgra[:,:,:3], gradient_bgra[:,:,3]
        blended_base = self._blend(gradient_bgr, base_image_bgr, blend_mode)
        alpha_norm = cv2.cvtColor(gradient_alpha.astype(np.float32) / 255, cv2.COLOR_GRAY2BGR)
        lit_bgr = (base_image_bgr * (1 - alpha_norm) + blended_base * alpha_norm).astype(np.uint8)

        if has_alpha:
            lit_image_with_alpha = cv2.cvtColor(lit_bgr, cv2.COLOR_BGR2BGRA)
            lit_image_with_alpha[:, :, 3] = original_alpha
            self.processed_img = self._apply_focus_effect(lit_image_with_alpha)
        else:
            self.processed_img = self._apply_focus_effect(lit_bgr)
        
        self.display_image(self.processed_img)

    def display_image(self, img_cv):
        if img_cv is None: return
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
            h, w, ch = img_cv.shape; qt_img = QImage(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA).data, w, h, ch * w, QImage.Format.Format_RGBA8888)
        else:
            h, w, ch = img_cv.shape; qt_img = QImage(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

if __name__ == '__main__':
    app = QApplication(sys.argv); ex = LightingSimulatorApp(); ex.show(); sys.exit(app.exec())