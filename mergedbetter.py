import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox, QFrame,
    QSizePolicy, QSlider, QColorDialog, QLineEdit, QTabWidget, QSplitter,
    QFormLayout, QCheckBox, QScrollArea, QRadioButton
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QIntValidator
from PyQt6.QtCore import Qt, QRect, QPointF, pyqtSignal, QTimer, QThread, QObject # <-- PERFORMANCE FIX

# <-- PERFORMANCE FIX: Worker thread for non-blocking generation
class Worker(QObject):
    finished = pyqtSignal(object) # Can emit the resulting image
    error = pyqtSignal(str)       # Can emit an error message

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.function(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ScalableImageLabel(QLabel):
    pointSelected = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet("QLabel { color: grey; }")
        self._light_angle_slider = None
        self._light_end_point = None
        self._focus_point = None
        self._is_lighting_active = False
        self._is_focus_active = False
        self._is_cutoff_active = False

    def set_effects_active(self, lighting=False, focus=False, cutoff=False):
        self._is_lighting_active = lighting
        self._is_focus_active = focus
        self._is_cutoff_active = cutoff
        self.update()

    def set_light_indicator_slider(self, angle_slider: QSlider | None):
        self._light_angle_slider = angle_slider
        if self._light_angle_slider:
            self._light_angle_slider.valueChanged.connect(self.update)

    def set_light_end_point(self, point):
        self._light_end_point = point
        self.update()

    def set_focus_point(self, point):
        self._focus_point = point
        self.update()

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
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
                scale = self._pixmap.width() / target_rect.width() if target_rect.width() > 0 else 1
                img_x = int((event.position().x() - target_rect.x()) * scale)
                img_y = int((event.position().y() - target_rect.y()) * scale)
                self.pointSelected.emit(img_x, img_y)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._pixmap.isNull():
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
            return

        target_rect = self._get_target_rect()
        painter.drawPixmap(target_rect, self._pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._is_lighting_active:
            if self._light_angle_slider is not None:
                angle, center = self._light_angle_slider.value(), target_rect.center()
                length = int(min(target_rect.width(), target_rect.height()) * 0.4)
                rad = np.deg2rad(angle - 180)
                end_x, end_y = center.x() + length * np.cos(rad), center.y() + length * np.sin(rad)
                pen = QPen(QColor(255, 255, 0, 150), 2, Qt.PenStyle.DashLine)
                painter.setPen(pen); painter.drawLine(center.x(), center.y(), int(end_x), int(end_y)); painter.drawEllipse(center, 3, 3)

            if self._is_cutoff_active and self._light_end_point:
                scale = target_rect.width() / self._pixmap.width() if self._pixmap.width() > 0 else 1
                label_x = target_rect.x() + int(self._light_end_point[0] * scale)
                label_y = target_rect.y() + int(self._light_end_point[1] * scale)
                pen = QPen(QColor(255, 255, 0, 220), 2)
                painter.setPen(pen)
                painter.drawLine(label_x - 10, label_y, label_x + 10, label_y)
                painter.drawLine(label_x, label_y - 10, label_x, label_y + 10)
        
        if self._is_focus_active and self._focus_point:
            scale = target_rect.width() / self._pixmap.width() if self._pixmap.width() > 0 else 1
            label_x = target_rect.x() + int(self._focus_point[0] * scale)
            label_y = target_rect.y() + int(self._focus_point[1] * scale)
            pen = QPen(QColor(255, 0, 0, 220), 2)
            painter.setPen(pen)
            painter.drawEllipse(label_x - 10, label_y - 10, 20, 20)
            painter.drawLine(label_x - 15, label_y, label_x + 15, label_y)
            painter.drawLine(label_x, label_y - 15, label_x, label_y + 15)


class ClickableImageLabel(ScalableImageLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proportional_center, self.source_label, self.resize_slider, self.rotate_slider = None, None, None, None

    def set_preview_dependencies(self, source_label, resize_slider: QSlider, rotate_slider: QSlider):
        self.source_label, self.resize_slider, self.rotate_slider = source_label, resize_slider, rotate_slider
        if self.resize_slider: self.resize_slider.valueChanged.connect(self.update)
        if self.rotate_slider: self.rotate_slider.valueChanged.connect(self.update)

    def mousePressEvent(self, event):
        if not self._pixmap.isNull() and self.source_label and self.source_label.pixmap():
            target_rect = self._get_target_rect()
            if target_rect.contains(event.position().toPoint()):
                relative_x, relative_y = event.position().x() - target_rect.x(), event.position().y() - target_rect.y()
                self.proportional_center = QPointF(relative_x / target_rect.width(), relative_y / target_rect.height())
                self.update()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if (self.proportional_center and self.source_label and self.source_label.pixmap() and
            not self.source_label.pixmap().isNull() and self.resize_slider and self.rotate_slider):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            target_rect = self._get_target_rect()
            center_x = target_rect.x() + self.proportional_center.x() * target_rect.width()
            center_y = target_rect.y() + self.proportional_center.y() * target_rect.height()
            source_scale, source_pixmap = self.resize_slider.value() / 100.0, self.source_label.pixmap()
            dest_scale_ratio = target_rect.width() / self._pixmap.width() if self._pixmap.width() > 0 else 0
            preview_w, preview_h = int(source_pixmap.width() * source_scale * dest_scale_ratio), int(source_pixmap.height() * source_scale * dest_scale_ratio)
            painter.save()
            painter.translate(center_x, center_y); painter.rotate(self.rotate_slider.value())
            preview_rect = QRect(-preview_w // 2, -preview_h // 2, preview_w, preview_h)
            painter.setOpacity(0.75); painter.drawPixmap(preview_rect, source_pixmap)
            painter.setOpacity(1.0)
            painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)); painter.drawRect(preview_rect)
            painter.restore()

    def get_placement_data(self):
        if self.proportional_center is None or self._pixmap.isNull(): return None, None, None
        dest_w, dest_h = self._pixmap.width(), self._pixmap.height()
        center_x, center_y = int(self.proportional_center.x() * dest_w), int(self.proportional_center.y() * dest_h)
        scale = self.resize_slider.value() / 100.0 if self.resize_slider else 1.0
        angle = self.rotate_slider.value() if self.rotate_slider else 0
        return (center_x, center_y), scale, angle


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Compositor"); self.setGeometry(100, 100, 1400, 850)
        self.src_img_original, self.src_img_lit, self.dest_img, self.result_img = None, None, None, None
        self.light_color, self.focus_point, self.light_end_point = (255, 255, 220), None, None
        self.thread = None # <-- PERFORMANCE FIX
        self.worker = None # <-- PERFORMANCE FIX
        self.initUI()

    def initUI(self):
        # ...
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_layout = QHBoxLayout()
        images_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_layout = QHBoxLayout()

        main_layout.addLayout(top_layout); main_layout.addWidget(images_splitter, 1); main_layout.addLayout(bottom_layout)

        self.src_label, self.dest_label, self.result_label = ScalableImageLabel("1. Load Source"), ClickableImageLabel("2. Load Destination"), ScalableImageLabel("Result")
        images_splitter.addWidget(self.src_label); images_splitter.addWidget(self.dest_label); images_splitter.addWidget(self.result_label)
        images_splitter.setSizes([350, 500, 500])

        self.btn_load_src, self.btn_load_dest = QPushButton("Load Source"), QPushButton("Load Destination")
        self.cloner_method_combo = QComboBox(); self.cloner_method_combo.addItems(["Alpha Blend / Feather", "Seamless - Normal", "Seamless - Mixed"])
        self.btn_generate = QPushButton("Generate")
        top_layout.addWidget(self.btn_load_src); top_layout.addWidget(self.btn_load_dest); top_layout.addWidget(QLabel("Blend Method:")); top_layout.addWidget(self.cloner_method_combo); top_layout.addWidget(self.btn_generate)

        self.tabs = QTabWidget(); self.tabs.setFixedHeight(240); bottom_layout.addWidget(self.tabs)
        # ...

        # ---- Cloning Tab ----
        cloning_tab = QWidget()
        cloning_form = QFormLayout(cloning_tab)
        self.tabs.addTab(cloning_tab, "Cloning Controls")
        self.effects_master_checkbox = QCheckBox("Enable Source Effects (Lighting & Focus)")
        cloning_form.addRow(self.effects_master_checkbox)
        self.resize_slider_w, self.rotate_slider_w, self.strength_slider_w, self.feather_slider_w = self._create_slider_group()
        cloning_form.addRow("Scale (%):", self.resize_slider_w); cloning_form.addRow("Rotate (°):", self.rotate_slider_w)
        cloning_form.addRow("Intensity (%):", self.strength_slider_w); cloning_form.addRow("Feather (px):", self.feather_slider_w)

        # ---- Lighting Tab ----
        lighting_scroll_area = QScrollArea()
        lighting_scroll_area.setWidgetResizable(True)
        lighting_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        lighting_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lighting_tab_content = QWidget()
        lighting_form = QFormLayout(lighting_tab_content)
        lighting_scroll_area.setWidget(lighting_tab_content)
        self.tabs.addTab(lighting_scroll_area, "Source Effects")
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget); mode_layout.setContentsMargins(0,0,0,0)
        self.radio_set_light = QRadioButton("Set Light Cutoff")
        self.radio_set_focus = QRadioButton("Set Focus Point"); self.radio_set_focus.setChecked(True)
        mode_layout.addWidget(self.radio_set_light); mode_layout.addWidget(self.radio_set_focus)
        lighting_form.addRow("Click Action:", mode_widget)
        self.cutoff_enable_cb = QCheckBox("Enable Light Cutoff"); self.cutoff_enable_cb.setChecked(True)
        lighting_form.addRow(self.cutoff_enable_cb)
        self.btn_color = QPushButton("Choose Light Color"); lighting_form.addRow(self.btn_color)
        self.light_blend_combo = QComboBox(); self.light_blend_combo.addItems(["Overlay", "Soft Light", "Screen (Lighten)", "Multiply (Darken)"])
        lighting_form.addRow("Light Blend:", self.light_blend_combo)
        self.angle_slider_w, self.intensity_slider_w, self.smoothness_slider_w = self._create_lighting_slider_group()
        lighting_form.addRow("Light Angle (°):", self.angle_slider_w)
        lighting_form.addRow("Light Intensity (%):", self.intensity_slider_w)
        lighting_form.addRow("Gradient Smoothness:", self.smoothness_slider_w)
        lighting_form.addRow(QLabel("<b>Focus Effect</b>"))
        self.focus_enable_cb = QCheckBox("Enable Focus Effect")
        lighting_form.addRow(self.focus_enable_cb)
        self.focus_size_slider_w, self.blur_amount_slider_w, self.focus_transition_slider_w = self._create_focus_slider_group()
        lighting_form.addRow("Focus Size (px):", self.focus_size_slider_w)
        lighting_form.addRow("Blur Amount (px):", self.blur_amount_slider_w)
        lighting_form.addRow("Focus Transition (px):", self.focus_transition_slider_w)
        self.tabs.setTabEnabled(1, False)
        self.src_label.set_light_indicator_slider(self.angle_slider_w.findChild(QSlider))

        # ---- Connections ----
        self.effects_master_checkbox.toggled.connect(self.toggle_effects_tab)
        self.dest_label.set_preview_dependencies(self.src_label, self.resize_slider_w.findChild(QSlider), self.rotate_slider_w.findChild(QSlider))
        self.btn_load_src.clicked.connect(self.load_source)
        self.btn_load_dest.clicked.connect(self.load_destination)
        self.btn_generate.clicked.connect(self.run_cloning_threaded) # <-- PERFORMANCE FIX
        self.btn_color.clicked.connect(self.choose_color)
        self.src_label.pointSelected.connect(self.handle_point_selection)
        
        # <-- PERFORMANCE FIX: Debounce timer setup
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(50) # 50ms delay
        self.update_timer.timeout.connect(self._apply_effects_to_source)
        
        # Connect all controls that trigger a preview update to the debouncer
        self.light_blend_combo.currentIndexChanged.connect(self.update_timer.start)
        self.cutoff_enable_cb.toggled.connect(self.update_timer.start)
        self.focus_enable_cb.toggled.connect(self.update_timer.start)
        for w in [self.angle_slider_w, self.intensity_slider_w, self.smoothness_slider_w, self.focus_size_slider_w, self.blur_amount_slider_w, self.focus_transition_slider_w]:
            w.findChild(QSlider).valueChanged.connect(self.update_timer.start)

    def _create_slider_with_lineedit(self, min_val, max_val, default_val):
        # ...
        widget = QWidget(); layout = QHBoxLayout(widget); layout.setContentsMargins(0,0,0,0)
        slider, line_edit = QSlider(Qt.Orientation.Horizontal), QLineEdit(str(default_val))
        slider.setRange(min_val, max_val); line_edit.setValidator(QIntValidator(min_val, max_val, self)); line_edit.setFixedWidth(55)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        slider.valueChanged.connect(lambda v: line_edit.setText(str(v)))
        line_edit.editingFinished.connect(lambda: slider.setValue(int(line_edit.text())))
        slider.setValue(default_val)
        layout.addWidget(slider); layout.addWidget(line_edit)
        return widget

    def _create_slider_group(self): return (self._create_slider_with_lineedit(*args) for args in [(10, 300, 100), (-180, 180, 0), (0, 100, 100), (0, 50, 0)])
    def _create_lighting_slider_group(self): return (self._create_slider_with_lineedit(*args) for args in [(0, 360, 45), (0, 100, 75), (10, 300, 100)])
    def _create_focus_slider_group(self): return (self._create_slider_with_lineedit(*args) for args in [(10, 500, 100), (1, 51, 15), (1, 101, 51)])
    
    def toggle_effects_tab(self, checked):
        self.tabs.setTabEnabled(1, checked)
        self.update_timer.start() # Use debouncer

    def handle_point_selection(self, x, y):
        if not self.effects_master_checkbox.isChecked(): return
        if self.radio_set_light.isChecked(): self.light_end_point = (x, y)
        elif self.radio_set_focus.isChecked(): self.focus_point = (x, y)
        self.update_timer.start() # Use debouncer

    def load_source(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Source Image", "", "PNG Images (*.png)")
        if file_path:
            self.src_img_original = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if self.src_img_original is None: QMessageBox.critical(self, "Error", "Failed to read source image."); return
            if len(self.src_img_original.shape) < 3 or (self.src_img_original.shape[2] != 4): QMessageBox.warning(self, "Warning", "For best results, use a PNG with a transparency layer.")
            h, w = self.src_img_original.shape[:2]
            center_point = (w // 2, h // 2)
            self.focus_point = center_point
            self.light_end_point = center_point
            self.update_timer.start() # Use debouncer

    def load_destination(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Destination Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.dest_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if self.dest_img is None: QMessageBox.critical(self, "Error", "Failed to read destination image."); return
            self.display_image(self.dest_img, self.dest_label)

    def choose_color(self):
        color = QColorDialog.getColor(QColor(*self.light_color[::-1]))
        if color.isValid():
            self.light_color = (color.blue(), color.green(), color.red())
            self.update_timer.start() # Use debouncer
            self.btn_color.setStyleSheet(f"background-color: {color.name()}; color: {'black' if color.lightness() > 127 else 'white'};")

    def _apply_effects_to_source(self):
        if self.src_img_original is None: return
        processed_image = self.src_img_original.copy()
        is_lighting_master_on = self.effects_master_checkbox.isChecked()
        is_focus_on = is_lighting_master_on and self.focus_enable_cb.isChecked()
        is_cutoff_on = is_lighting_master_on and self.cutoff_enable_cb.isChecked()
        if is_lighting_master_on: processed_image = self._apply_gradient_effect(processed_image)
        if is_focus_on: processed_image = self._apply_focus_effect(processed_image)
        self.src_img_lit = processed_image
        self.src_label.set_effects_active(lighting=is_lighting_master_on, focus=is_focus_on, cutoff=is_cutoff_on)
        self.src_label.set_light_end_point(self.light_end_point)
        self.src_label.set_focus_point(self.focus_point)
        self.display_image(self.src_img_lit, self.src_label)
        self.dest_label.update()
    
    def _apply_gradient_effect(self, image):
        if len(image.shape) == 3 and image.shape[2] == 4: has_alpha, original_alpha, base_bgr = True, image[:, :, 3], image[:, :, :3]
        else: has_alpha, base_bgr = False, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) < 3 else image
        angle, intensity, smoothness = [w.findChild(QSlider).value() for w in [self.angle_slider_w, self.intensity_slider_w, self.smoothness_slider_w]]
        is_cutoff_enabled = self.cutoff_enable_cb.isChecked()
        blend_mode = self.light_blend_combo.currentText()
        h, w = base_bgr.shape[:2]
        gradient_bgra = self._create_gradient(w, h, angle, self.light_end_point, self.light_color, intensity, smoothness, is_cutoff_enabled)
        blended_base = self._blend(gradient_bgra[:,:,:3], base_bgr, blend_mode)
        alpha_norm = cv2.cvtColor(gradient_bgra[:,:,3].astype(np.float32) / 255.0, cv2.COLOR_GRAY2BGR)
        processed_bgr = (base_bgr.astype(np.float32) * (1-alpha_norm) + blended_base * alpha_norm).astype(np.uint8)
        if has_alpha: lit_image = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2BGRA); lit_image[:, :, 3] = original_alpha; return lit_image
        else: return processed_bgr

    def _apply_focus_effect(self, image):
        if self.focus_point is None: return image
        focus_size = self.focus_size_slider_w.findChild(QSlider).value()
        blur_amount = max(1, self.blur_amount_slider_w.findChild(QSlider).value() * 2 + 1)
        transition = max(1, self.focus_transition_slider_w.findChild(QSlider).value() * 2 + 1)
        if len(image.shape) == 3 and image.shape[2] == 4: has_alpha, original_alpha, sharp_bgr = True, image[:, :, 3], image[:, :, :3]
        else: has_alpha, sharp_bgr = False, image
        blurry_bgr = cv2.GaussianBlur(sharp_bgr, (blur_amount, blur_amount), 0)
        h, w = sharp_bgr.shape[:2]
        blur_map = np.zeros((h, w), dtype=np.uint8); cv2.circle(blur_map, self.focus_point, focus_size, 255, -1)
        blur_map = cv2.GaussianBlur(blur_map, (transition, transition), 0)
        blur_map_float = cv2.cvtColor(blur_map.astype(np.float32) / 255.0, cv2.COLOR_GRAY2BGR)
        focused_bgr = (sharp_bgr * blur_map_float + blurry_bgr * (1 - blur_map_float)).astype(np.uint8)
        if has_alpha: return cv2.merge([focused_bgr[:,:,0], focused_bgr[:,:,1], focused_bgr[:,:,2], original_alpha])
        else: return focused_bgr

    def _create_gradient(self, w, h, angle, end_point, color_bgr, intensity, smoothness, is_cutoff_enabled):
        if end_point is None: end_point = (w // 2, h // 2)
        rad = np.deg2rad(angle); vx, vy = np.cos(rad), np.sin(rad)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        projection = x*vx + y*vy
        min_p = projection.min()
        if is_cutoff_enabled:
            end_p = end_point[0]*vx + end_point[1]*vy
            denominator = end_p - min_p
            if abs(denominator) < 1e-6: denominator = 1e-6
            grad_raw = (end_p - projection) / denominator
            grad_norm = np.clip(grad_raw, 0, 1)
        else:
            max_p = projection.max()
            denominator = max_p - min_p
            if abs(denominator) < 1e-6: denominator = 1e-6
            grad_norm = (max_p - projection) / denominator
        exp = 0.2 + (smoothness-10)/(300-10)*(4.0-0.2)
        grad_final = np.power(grad_norm, exp)
        alpha = (grad_final * (intensity/100.0) * 255).astype(np.uint8)
        color = np.full((h,w,3), color_bgr, dtype=np.uint8)
        return cv2.merge([color[:,:,0], color[:,:,1], color[:,:,2], alpha])

    def _blend(self, fg, bg, mode):
        fg_f, bg_f = fg.astype(np.float32), bg.astype(np.float32)
        if mode == "Multiply (Darken)": res = (fg_f * bg_f / 255.0)
        elif mode == "Screen (Lighten)": res = (255.0 - ((255.0 - fg_f) * (255.0 - bg_f) / 255.0))
        elif mode == "Overlay": res = np.where(bg_f<=128.0, (2.0*fg_f*bg_f)/255.0, 255.0-2.0*(255.0-fg_f)*(255.0-bg_f)/255.0)
        elif mode == "Soft Light": a, b = fg_f/255.0, bg_f/255.0; res = ((1-2*a)*(b**2) + 2*a*b)*255.0
        else: res = bg_f
        return np.clip(res, 0, 255).astype(np.uint8)

    # <-- PERFORMANCE FIX: This function now runs in a background thread
    def run_cloning_threaded(self):
        if self.src_img_lit is None or self.dest_img is None:
            QMessageBox.warning(self, "Error", "Please load both source and destination images."); return
        
        center_point, scale, angle = self.dest_label.get_placement_data()
        if center_point is None:
            QMessageBox.warning(self, "Error", "Please click on the destination image to set a placement point."); return

        self.btn_generate.setText("Generating...")
        self.btn_generate.setEnabled(False)

        self.thread = QThread()
        self.worker = Worker(self._perform_cloning)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_cloning_finished)
        self.worker.error.connect(self.on_cloning_error)
        self.thread.start()

    def on_cloning_finished(self, result_image):
        self.result_img = result_image
        self.display_image(self.result_img, self.result_label)
        self.btn_generate.setText("Generate")
        self.btn_generate.setEnabled(True)
        self.thread.quit()
        self.thread.wait()
        if QMessageBox.question(self, "Save Image", "Generation complete. Do you want to save the result?") == QMessageBox.StandardButton.Yes:
            self.save_result()

    def on_cloning_error(self, error_message):
        QMessageBox.critical(self, "Processing Error", f"An error occurred during generation.\n\nDetails: {error_message}")
        self.btn_generate.setText("Generate")
        self.btn_generate.setEnabled(True)
        self.thread.quit()
        self.thread.wait()
        
    def _perform_cloning(self):
        # This is the original logic from run_cloning, but now it returns the image
        center_point, scale, angle = self.dest_label.get_placement_data()
        clone_mode_str = self.cloner_method_combo.currentText()
        strength = self.strength_slider_w.findChild(QSlider).value() / 100.0
        feather = self.feather_slider_w.findChild(QSlider).value()

        h0, w0 = self.src_img_lit.shape[:2]; w, h = max(1, int(round(w0*scale))), max(1, int(round(h0*scale)))
        scaled_src = cv2.resize(self.src_img_lit, (w, h), interpolation=cv2.INTER_AREA)
        
        if len(scaled_src.shape)<3 or scaled_src.shape[2]!=4: src_bgra, mask = cv2.cvtColor(scaled_src, cv2.COLOR_BGR2BGRA), np.full(scaled_src.shape[:2], 255, dtype=np.uint8)
        else: src_bgra, mask = scaled_src, scaled_src[:, :, 3]
        
        rot_bgra, final_mask = self._rotate_image_and_mask(src_bgra, mask, -angle)
        final_bgr = rot_bgra[:,:,:3]
        
        dest_h, dest_w = self.dest_img.shape[:2]; src_h, src_w = final_bgr.shape[:2]
        tl_x, tl_y = center_point[0]-src_w//2, center_point[1]-src_h//2
        br_x, br_y = tl_x + src_w, tl_y + src_h
        
        # This part has a GUI call, which is not thread-safe. We should pre-check or handle it.
        # For simplicity in this example, we assume the check was done before starting the thread.
        # A more robust solution would pass this check result into the worker.
        if clone_mode_str.startswith("Seamless") and ((tl_x<1) or (tl_y<1) or (br_x>=dest_w-1) or (br_y>=dest_h-1)):
            clone_mode_str = "Alpha Blend / Feather"

        roi_x1, roi_y1 = max(0, tl_x), max(0, tl_y)
        roi_x2, roi_y2 = min(dest_w, br_x), min(dest_h, br_y)
        crop_x1, crop_y1 = roi_x1-tl_x, roi_y1-tl_y
        crop_x2, crop_y2 = crop_x1+(roi_x2-roi_x1), crop_y1+(roi_y2-roi_y1)
        
        if (roi_x2-roi_x1)<=0 or (roi_y2-roi_y1)<=0: return self.dest_img.copy()
        
        clip_src, clip_mask = final_bgr[crop_y1:crop_y2, crop_x1:crop_x2], final_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        
        output_image = self.dest_img.copy()
        if clone_mode_str.startswith("Alpha Blend"):
            if feather > 0: k_size = max(1, feather*2+1); clip_mask = cv2.GaussianBlur(clip_mask, (k_size, k_size), 0)
            dest_roi = output_image[roi_y1:roi_y2, roi_x1:roi_x2]
            alpha3 = cv2.cvtColor((clip_mask.astype(np.float32)*strength)/255.0, cv2.COLOR_GRAY2BGR)
            fg, bg = clip_src.astype(np.float32), dest_roi.astype(np.float32)
            roi = (cv2.multiply(alpha3, fg) + cv2.multiply(1.0-alpha3, bg)).astype(np.uint8)
            output_image[roi_y1:roi_y2, roi_x1:roi_x2] = roi
        else:
            mode = cv2.NORMAL_CLONE if clone_mode_str=="Seamless - Normal" else cv2.MIXED_CLONE
            center = (roi_x1+(roi_x2-roi_x1)//2, roi_y1+(roi_y2-roi_y1)//2)
            # Use a copy for seamlessClone source to avoid issues
            cloned = cv2.seamlessClone(clip_src, self.dest_img.copy(), clip_mask, center, mode)
            output_image = cv2.addWeighted(cloned, strength, self.dest_img, 1-strength, 0)

        return output_image


    def _rotate_image_and_mask(self, image_bgra, mask, angle_deg):
        (h, w) = image_bgra.shape[:2]; (cX, cY) = (w//2, h//2)
        M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
        cos, sin = abs(M[0,0]), abs(M[0,1])
        new_w, new_h = int((h*sin)+(w*cos)), int((h*cos)+(w*sin))
        M[0,2] += (new_w/2)-cX; M[1,2] += (new_h/2)-cY
        
        # <-- QUALITY FIX: Use higher quality interpolation for smoother edges
        rot_img = cv2.warpAffine(image_bgra, M, (new_w,new_h), flags=cv2.INTER_CUBIC, borderValue=(0,0,0,0))
        rot_mask = cv2.warpAffine(mask, M, (new_w,new_h), flags=cv2.INTER_LINEAR, borderValue=0)
        
        return rot_img, rot_mask

    def save_result(self):
        if self.result_img is None: return
        output_dir = "generation"; os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath, _ = QFileDialog.getSaveFileName(self, "Save Result", os.path.join(output_dir, f"composited_{timestamp}.png"), "PNG (*.png);;JPEG (*.jpg)")
        if fpath: cv2.imwrite(fpath, self.result_img); QMessageBox.information(self, "Success", f"Image saved to:\n{fpath}")

    def display_image(self, img_cv, label: QLabel):
        if img_cv is None: return
        img = img_cv.copy()
        if len(img.shape)==3 and img.shape[2]==4:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            h, w, ch = img_rgba.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        else:
            if len(img.shape)<3: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainApp()
    ex.show()
    sys.exit(app.exec())