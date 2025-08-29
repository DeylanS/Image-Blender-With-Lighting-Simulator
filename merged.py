import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox, QFrame,
    QSizePolicy, QSlider, QColorDialog, QLineEdit, QTabWidget, QSplitter,
    QFormLayout, QCheckBox
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QIntValidator
from PyQt6.QtCore import Qt, QRect, QPointF


# =========================
#  Scalable Image Label(s)
# =========================
class ScalableImageLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet("QLabel { color: grey; }")

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def pixmap(self):
        return self._pixmap

    def _get_target_rect(self):
        if self._pixmap.isNull():
            return QRect()
        label_size = self.size()
        scaled_pixmap = self._pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (label_size.width() - scaled_pixmap.width()) / 2
        y = (label_size.height() - scaled_pixmap.height()) / 2
        return QRect(int(x), int(y), scaled_pixmap.width(), scaled_pixmap.height())

    def paintEvent(self, event):
        if self._pixmap.isNull():
            painter = QPainter(self)
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
            return
        target_rect = self._get_target_rect()
        painter = QPainter(self)
        painter.drawPixmap(target_rect, self._pixmap)

class ClickableImageLabel(ScalableImageLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proportional_center = None
        self.source_label = None
        self.resize_slider = None
        self.rotate_slider = None

    def set_preview_dependencies(self, source_label, resize_slider, rotate_slider):
        self.source_label = source_label
        self.resize_slider = resize_slider
        self.rotate_slider = rotate_slider
        if self.resize_slider:
            self.resize_slider.valueChanged.connect(self.update)
        if self.rotate_slider:
            self.rotate_slider.valueChanged.connect(self.update)

    def mousePressEvent(self, event):
        if not self._pixmap.isNull() and self.source_label and self.source_label.pixmap():
            target_rect = self._get_target_rect()
            if target_rect.contains(event.position().toPoint()):
                relative_x = event.position().x() - target_rect.x()
                relative_y = event.position().y() - target_rect.y()
                self.proportional_center = QPointF(
                    relative_x / target_rect.width(),
                    relative_y / target_rect.height()
                )
                self.update()
                super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.proportional_center and self.source_label and not self.source_label.pixmap().isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            target_rect = self._get_target_rect()
            center_x = target_rect.x() + self.proportional_center.x() * target_rect.width()
            center_y = target_rect.y() + self.proportional_center.y() * target_rect.height()
            source_scale = self.resize_slider.value() / 100.0
            source_pixmap = self.source_label.pixmap()
            dest_scale_ratio = target_rect.width() / self._pixmap.width() if self._pixmap.width() > 0 else 0
            preview_w = int(source_pixmap.width() * source_scale * dest_scale_ratio)
            preview_h = int(source_pixmap.height() * source_scale * dest_scale_ratio)
            painter.save()
            painter.translate(center_x, center_y)
            painter.rotate(self.rotate_slider.value())
            preview_rect = QRect(-preview_w // 2, -preview_h // 2, preview_w, preview_h)
            painter.setOpacity(0.75)
            painter.drawPixmap(preview_rect, source_pixmap)
            painter.setOpacity(1.0)
            painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine))
            painter.drawRect(preview_rect)
            painter.restore()

    def get_placement_data(self):
        if self.proportional_center is None or self._pixmap.isNull():
            return None, None, None
        dest_w = self._pixmap.width()
        dest_h = self._pixmap.height()
        center_x = int(self.proportional_center.x() * dest_w)
        center_y = int(self.proportional_center.y() * dest_h)
        scale = self.resize_slider.value() / 100.0
        angle = self.rotate_slider.value()
        return (center_x, center_y), scale, angle

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Compositor")
        self.setGeometry(100, 100, 1400, 850)
        self.src_img_original = None
        self.src_img_lit = None
        self.dest_img = None
        self.result_img = None
        self.light_color = (255, 255, 220)
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_layout = QHBoxLayout()
        images_splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- NEW: Container for bottom controls to manage sizing ---
        bottom_controls_container = QWidget()
        bottom_controls_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bottom_layout = QHBoxLayout(bottom_controls_container)
        bottom_layout.setContentsMargins(0,0,0,0)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(images_splitter, 1)
        main_layout.addWidget(bottom_controls_container)

        self.src_label = ScalableImageLabel("1. Load Source Image")
        self.dest_label = ClickableImageLabel("2. Load Destination Image")
        self.result_label = ScalableImageLabel("Result")
        for w in [self.src_label, self.dest_label, self.result_label]:
            images_splitter.addWidget(w)
        images_splitter.setSizes([350, 500, 500])

        self.btn_load_src = QPushButton("Load Source")
        self.btn_load_dest = QPushButton("Load Destination")
        self.cloner_method_combo = QComboBox()
        self.cloner_method_combo.addItems(["Alpha Blend / Feather", "Seamless - Normal", "Seamless - Mixed"])
        self.btn_generate = QPushButton("Generate")

        top_layout.addWidget(self.btn_load_src)
        top_layout.addWidget(self.btn_load_dest)
        top_layout.addWidget(QLabel("Blend Method:"))
        top_layout.addWidget(self.cloner_method_combo)
        top_layout.addWidget(self.btn_generate)

        self.tabs = QTabWidget()
        bottom_layout.addWidget(self.tabs)
        self.tabs.setFixedHeight(220)

        cloning_tab = QWidget()
        cloning_form_layout = QFormLayout(cloning_tab)
        self.tabs.addTab(cloning_tab, "Cloning Controls")

        # --- NEW: Checkbox to enable/disable lighting ---
        self.lighting_checkbox = QCheckBox("Enable Source Lighting")
        cloning_form_layout.addRow(self.lighting_checkbox)

        self.resize_slider, self.rotate_slider, self.strength_slider, self.feather_slider = self._create_slider_group()
        cloning_form_layout.addRow("Scale (%):", self.resize_slider)
        cloning_form_layout.addRow("Rotate (°):", self.rotate_slider)
        cloning_form_layout.addRow("Intensity (%):", self.strength_slider)
        cloning_form_layout.addRow("Feather (px):", self.feather_slider)

        lighting_tab = QWidget()
        lighting_form_layout = QFormLayout(lighting_tab)
        self.tabs.addTab(lighting_tab, "Source Lighting")

        self.btn_color = QPushButton("Choose Light Color")
        self.light_blend_combo = QComboBox()
        self.light_blend_combo.addItems(["Overlay", "Soft Light", "Screen (Lighten)", "Multiply (Darken)"])
        self.angle_slider, self.intensity_slider, self.smoothness_slider = self._create_lighting_slider_group()

        lighting_form_layout.addRow(self.btn_color)
        lighting_form_layout.addRow("Light Blend:", self.light_blend_combo)
        lighting_form_layout.addRow("Light Angle (°):", self.angle_slider)
        lighting_form_layout.addRow("Light Intensity (%):", self.intensity_slider)
        lighting_form_layout.addRow("Gradient Smoothness:", self.smoothness_slider)

        # --- NEW: Disable lighting tab by default and connect checkbox ---
        self.tabs.setTabEnabled(1, False)
        self.lighting_checkbox.toggled.connect(self.toggle_lighting_tab)

        self.dest_label.set_preview_dependencies(
            self.src_label,
            self.resize_slider.findChild(QSlider),
            self.rotate_slider.findChild(QSlider)
        )

        self.btn_load_src.clicked.connect(self.load_source)
        self.btn_load_dest.clicked.connect(self.load_destination)
        self.btn_generate.clicked.connect(self.run_cloning)

        self.btn_color.clicked.connect(self.choose_color)
        self.light_blend_combo.currentIndexChanged.connect(self._apply_lighting_to_source)
        for slider_widget in [self.angle_slider, self.intensity_slider, self.smoothness_slider]:
            slider_widget.findChild(QSlider).valueChanged.connect(self._apply_lighting_to_source)

    def toggle_lighting_tab(self, checked):
        """Enables or disables the lighting tab and updates the source image."""
        self.tabs.setTabEnabled(1, checked)
        self._apply_lighting_to_source() # Re-apply/remove lighting

    def _create_slider_with_lineedit(self, min_val, max_val, default_val):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        line_edit = QLineEdit(str(default_val))
        slider.setRange(min_val, max_val)
        line_edit.setValidator(QIntValidator(min_val, max_val, self))
        line_edit.setFixedWidth(45)
        slider.valueChanged.connect(lambda v: line_edit.setText(str(v)))
        def _sync_from_edit():
            txt = line_edit.text()
            if txt == "": line_edit.setText(str(slider.value())); return
            slider.setValue(int(txt))
        line_edit.editingFinished.connect(_sync_from_edit)
        slider.setValue(default_val)
        layout.addWidget(slider)
        layout.addWidget(line_edit)
        return widget

    def _create_slider_group(self):
        return (self._create_slider_with_lineedit(*args) for args in [(10, 200, 100), (-180, 180, 0), (0, 100, 100), (0, 50, 0)])

    def _create_lighting_slider_group(self):
        return (self._create_slider_with_lineedit(*args) for args in [(0, 360, 45), (0, 100, 75), (10, 300, 100)])

    def load_source(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Source Image", "", "PNG Images (*.png)")
        if file_path:
            self.src_img_original = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if self.src_img_original is None: QMessageBox.critical(self, "Error", "Failed to read source image."); return
            if len(self.src_img_original.shape) < 3 or (self.src_img_original.shape[2] != 4): QMessageBox.warning(self, "Warning", "For best results, use a PNG with a transparency layer.")
            self._apply_lighting_to_source()

    def load_destination(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Destination Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.dest_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if self.dest_img is None: QMessageBox.critical(self, "Error", "Failed to read destination image."); return
            self.display_image(self.dest_img, self.dest_label)

    def _apply_lighting_to_source(self):
        if self.src_img_original is None: return

        if not self.lighting_checkbox.isChecked():
            self.src_img_lit = self.src_img_original.copy()
            self.display_image(self.src_img_lit, self.src_label)
            self.dest_label.update()
            return

        if len(self.src_img_original.shape) == 3 and self.src_img_original.shape[2] == 4:
            has_alpha, original_alpha, base_image_bgr = True, self.src_img_original[:, :, 3], self.src_img_original[:, :, :3]
        else:
            has_alpha, base_image_bgr = False, cv2.cvtColor(self.src_img_original, cv2.COLOR_GRAY2BGR) if len(self.src_img_original.shape) < 3 else self.src_img_original

        angle = self.angle_slider.findChild(QSlider).value()
        intensity = self.intensity_slider.findChild(QSlider).value()
        smoothness = self.smoothness_slider.findChild(QSlider).value()
        blend_mode = self.light_blend_combo.currentText()
        h, w = base_image_bgr.shape[:2]

        gradient_bgra = self._create_gradient(w, h, angle, self.light_color, intensity, smoothness)
        gradient_bgr, gradient_alpha = gradient_bgra[:, :, :3], gradient_bgra[:, :, 3]

        blended_base = self._blend(gradient_bgr, base_image_bgr, blend_mode)

        alpha_norm = cv2.cvtColor(gradient_alpha.astype(np.float32) / 255.0, cv2.COLOR_GRAY2BGR)
        processed_bgr = (base_image_bgr.astype(np.float32) * (1 - alpha_norm) + blended_base * alpha_norm).astype(np.uint8)

        if has_alpha:
            self.src_img_lit = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2BGRA)
            self.src_img_lit[:, :, 3] = original_alpha
        else:
            self.src_img_lit = processed_bgr

        self.display_image(self.src_img_lit, self.src_label)
        self.dest_label.update()

    def _create_gradient(self, w, h, angle, color_bgr, intensity, smoothness):
        rad = np.deg2rad(angle); vx, vy = np.cos(rad), np.sin(rad)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        projection = x * vx + y * vy
        min_proj, max_proj = projection.min(), projection.max()
        if max_proj == min_proj: gradient_norm = np.zeros((h, w), dtype=np.float32)
        else: gradient_norm = (projection - min_proj) / (max_proj - min_proj)
        exponent = 0.2 + (smoothness - 10) / (300 - 10) * (4.0 - 0.2)
        alpha_channel = (gradient_norm ** exponent * (intensity / 100.0) * 255).astype(np.uint8)
        color_layer = np.full((h, w, 3), color_bgr, dtype=np.uint8)
        return cv2.merge([color_layer[:, :, 0], color_layer[:, :, 1], color_layer[:, :, 2], alpha_channel])

    def _blend(self, fg, bg, mode):
        fg_float, bg_float = fg.astype(np.float32), bg.astype(np.float32)
        if mode == "Multiply (Darken)": result = (fg_float * bg_float / 255.0)
        elif mode == "Screen (Lighten)": result = (255.0 - ((255.0 - fg_float) * (255.0 - bg_float) / 255.0))
        elif mode == "Overlay": result = np.where(bg_float <= 128.0, (2.0 * fg_float * bg_float) / 255.0, 255.0 - 2.0 * (255.0 - fg_float) * (255.0 - bg_float) / 255.0)
        elif mode == "Soft Light":
            a, b = fg_float / 255.0, bg_float / 255.0
            result = ((1 - 2 * a) * (b ** 2) + 2 * a * b) * 255.0
        else: result = bg_float
        return np.clip(result, 0, 255).astype(np.uint8)

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.light_color = (color.blue(), color.green(), color.red())
            self._apply_lighting_to_source()
            self.btn_color.setStyleSheet(f"background-color: {color.name()}; color: {'black' if color.lightness() > 127 else 'white'};")

    def run_cloning(self):
        center_point, scale, angle = self.dest_label.get_placement_data()
        clone_mode_str = self.cloner_method_combo.currentText()
        strength = self.strength_slider.findChild(QSlider).value() / 100.0
        feather_amount = self.feather_slider.findChild(QSlider).value()

        if self.src_img_lit is None or self.dest_img is None or center_point is None:
            QMessageBox.warning(self, "Error", "Please load images and click a destination point."); return

        try:
            src_h0, src_w0 = self.src_img_lit.shape[:2]
            new_w, new_h = max(1, int(round(src_w0*scale))), max(1, int(round(src_h0*scale)))
            scaled_src = cv2.resize(self.src_img_lit, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if len(scaled_src.shape) < 3 or scaled_src.shape[2] != 4:
                src_bgra, mask = cv2.cvtColor(scaled_src, cv2.COLOR_BGR2BGRA), np.full(scaled_src.shape[:2], 255, dtype=np.uint8)
            else: src_bgra, mask = scaled_src, scaled_src[:, :, 3]
            
            rotated_src_bgra, final_mask = self._rotate_image_and_mask(src_bgra, mask, -angle)
            final_src_bgr = rotated_src_bgra[:,:,:3]
            
            dest_h, dest_w = self.dest_img.shape[:2]
            src_h, src_w = final_src_bgr.shape[:2]
            top_left_x, top_left_y = center_point[0] - src_w // 2, center_point[1] - src_h // 2
            bottom_right_x, bottom_right_y = top_left_x + src_w, top_left_y + src_h
            
            is_out_of_bounds = (top_left_x < 1) or (top_left_y < 1) or (bottom_right_x >= dest_w-1) or (bottom_right_y >= dest_h-1)
            if is_out_of_bounds and clone_mode_str.startswith("Seamless"):
                QMessageBox.information(self, "Mode Switched", "Seamless Clone requires a border and fails at the image edge.\nSwitching to 'Alpha Blend' for this generation.")
                clone_mode_str = "Alpha Blend / Feather"

            dest_roi_x1, dest_roi_y1 = max(0, top_left_x), max(0, top_left_y)
            dest_roi_x2, dest_roi_y2 = min(dest_w, bottom_right_x), min(dest_h, bottom_right_y)
            src_crop_x1, src_crop_y1 = dest_roi_x1 - top_left_x, dest_roi_y1 - top_left_y
            src_crop_x2, src_crop_y2 = src_crop_x1 + (dest_roi_x2 - dest_roi_x1), src_crop_y1 + (dest_roi_y2 - dest_roi_y1)
            
            if (dest_roi_x2 - dest_roi_x1) <= 0 or (dest_roi_y2 - dest_roi_y1) <= 0: return

            clipped_src = final_src_bgr[src_crop_y1:src_crop_y2, src_crop_x1:src_crop_x2]
            clipped_mask = final_mask[src_crop_y1:src_crop_y2, src_crop_x1:src_crop_x2]
            
            if clone_mode_str.startswith("Alpha Blend"):
                if feather_amount > 0:
                    k_size = max(1, feather_amount * 2 + 1)
                    clipped_mask = cv2.GaussianBlur(clipped_mask, (k_size, k_size), 0)
                
                dest_roi = self.dest_img[dest_roi_y1:dest_roi_y2, dest_roi_x1:dest_roi_x2]
                alpha_3ch = cv2.cvtColor((clipped_mask.astype(np.float32) * strength) / 255.0, cv2.COLOR_GRAY2BGR)
                fg, bg = clipped_src.astype(np.float32), dest_roi.astype(np.float32)
                blended_roi = (cv2.multiply(alpha_3ch, fg) + cv2.multiply(1.0 - alpha_3ch, bg)).astype(np.uint8)
                self.result_img = self.dest_img.copy()
                self.result_img[dest_roi_y1:dest_roi_y2, dest_roi_x1:dest_roi_x2] = blended_roi
            else: # Seamless Modes
                mode = cv2.NORMAL_CLONE if clone_mode_str == "Seamless - Normal" else cv2.MIXED_CLONE
                safe_center = (dest_roi_x1 + (dest_roi_x2 - dest_roi_x1) // 2, dest_roi_y1 + (dest_roi_y2 - dest_roi_y1) // 2)
                cloned_at_100 = cv2.seamlessClone(clipped_src, self.dest_img.copy(), clipped_mask, safe_center, mode)
                self.result_img = cv2.addWeighted(cloned_at_100, strength, self.dest_img, 1 - strength, 0)

            self.display_image(self.result_img, self.result_label)
            if QMessageBox.question(self, "Save Image", "Do you want to save the result?") == QMessageBox.StandardButton.Yes: self.save_result()
        except (cv2.error, ValueError) as e:
            QMessageBox.critical(self, "Processing Error", f"An error occurred.\n\nDetails: {e}")

    def _rotate_image_and_mask(self, image_bgra, mask, angle_deg):
        (h, w) = image_bgra.shape[:2]; (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - cX; M[1, 2] += (new_h / 2) - cY
        rotated_image = cv2.warpAffine(image_bgra, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
        rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
        return rotated_image, rotated_mask

    def save_result(self):
        if self.result_img is None: return
        output_dir = "generation"; os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Result", os.path.join(output_dir, f"composited_{timestamp}.png"), "PNG (*.png);;JPEG (*.jpg)")
        if file_path: cv2.imwrite(file_path, self.result_img); QMessageBox.information(self, "Success", f"Image saved to:\n{file_path}")

    def display_image(self, img_cv, label: QLabel):
        if img_cv is None: return
        img = img_cv.copy()
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            h, w = img_rgba.shape[:2]
            qimg = QImage(img_rgba.data, w, h, img_rgba.strides[0], QImage.Format.Format_RGBA8888)
        else:
            if len(img.shape) < 3: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            qimg = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainApp()
    ex.show()
    sys.exit(app.exec())