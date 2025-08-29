import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox, QFrame,
    QSizePolicy, QSlider
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt6.QtCore import Qt, QPoint, QRect, QSize, QPointF

# (ScalableImageLabel and ClickableImageLabel classes are unchanged)
class ScalableImageLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pixmap = QPixmap()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 300)
        self.setFrameShape(QFrame.Shape.Box)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #f0f0f0; color: grey; }")

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

    def paintEvent(self, event):
        if self._pixmap.isNull():
            super().paintEvent(event)
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

    def set_preview_dependencies(self, source_label, slider):
        self.source_label = source_label
        self.resize_slider = slider
        if self.resize_slider: self.resize_slider.valueChanged.connect(self.update)

    def mousePressEvent(self, event):
        if not self._pixmap.isNull() and self.source_label and self.source_label.pixmap():
            target_rect = self._get_target_rect()
            if target_rect.contains(event.position().toPoint()):
                relative_x = event.position().x() - target_rect.x()
                relative_y = event.position().y() - target_rect.y()
                prop_x = relative_x / target_rect.width()
                prop_y = relative_y / target_rect.height()
                self.proportional_center = QPointF(prop_x, prop_y)
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
            dest_scale_ratio = target_rect.width() / self._pixmap.width()
            preview_w = int(source_pixmap.width() * source_scale * dest_scale_ratio)
            preview_h = int(source_pixmap.height() * source_scale * dest_scale_ratio)
            px = int(center_x - preview_w // 2)
            py = int(center_y - preview_h // 2)
            preview_rect = QRect(px, py, preview_w, preview_h)
            painter.setOpacity(0.75)
            painter.drawPixmap(preview_rect, source_pixmap)
            painter.setOpacity(1.0)
            pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(preview_rect)

    def get_placement_data(self):
        if self.proportional_center is None or self._pixmap.isNull(): return None, None
        dest_w, dest_h = self._pixmap.width(), self._pixmap.height()
        center_x = int(self.proportional_center.x() * dest_w)
        center_y = int(self.proportional_center.y() * dest_h)
        scale = self.resize_slider.value() / 100.0
        return (center_x, center_y), scale

class SeamlessClonerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Seamless Cloner")
        self.setGeometry(100, 100, 1200, 750)
        self.src_img, self.dest_img, self.result_img = None, None, None
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        controls_layout, images_layout = QHBoxLayout(), QHBoxLayout()
        sliders_layout = QVBoxLayout()

        self.btn_load_src = QPushButton("1. Load Source (PNG)")
        self.btn_load_dest = QPushButton("2. Load Destination")
        self.clone_method_combo = QComboBox()
        
        # --- NEW: Add a full range of artistic blend modes ---
        self.clone_method_combo.addItem("Normal (Alpha Blend)")
        self.clone_method_combo.insertSeparator(1)
        self.clone_method_combo.addItem("Seamless - Normal")
        self.clone_method_combo.addItem("Seamless - Mixed")
        self.clone_method_combo.insertSeparator(4)
        self.clone_method_combo.addItem("Multiply (Darken)")
        self.clone_method_combo.addItem("Screen (Lighten)")
        self.clone_method_combo.addItem("Overlay (Contrast)")
        self.clone_method_combo.addItem("Soft Light (Subtle)")

        self.btn_generate = QPushButton("5. Generate")

        controls_layout.addWidget(self.btn_load_src)
        controls_layout.addWidget(self.btn_load_dest)
        controls_layout.addWidget(QLabel("3. Blend Method:"))
        controls_layout.addWidget(self.clone_method_combo)
        controls_layout.addWidget(self.btn_generate)

        self.src_label = ScalableImageLabel("Source Image")
        self.dest_label = ClickableImageLabel("Destination Image\n(Click to place)")
        self.result_label = ScalableImageLabel("Result")
        images_layout.addWidget(self.src_label)
        images_layout.addWidget(self.dest_label)
        images_layout.addWidget(self.result_label)

        scale_slider_layout = QHBoxLayout()
        self.resize_slider = QSlider(Qt.Orientation.Horizontal)
        self.resize_slider.setRange(10, 200); self.resize_slider.setValue(100)
        self.scale_label = QLabel(f"Scale: {self.resize_slider.value()}%")
        self.resize_slider.valueChanged.connect(lambda v: self.scale_label.setText(f"Scale: {v}%"))
        scale_slider_layout.addWidget(QLabel("4. Controls:"))
        scale_slider_layout.addWidget(self.resize_slider)
        scale_slider_layout.addWidget(self.scale_label)

        strength_slider_layout = QHBoxLayout()
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100); self.strength_slider.setValue(100)
        self.strength_label = QLabel(f"Intensity: {self.strength_slider.value()}%")
        self.strength_slider.valueChanged.connect(lambda v: self.strength_label.setText(f"Intensity: {v}%"))
        strength_slider_layout.addWidget(self.strength_label)
        strength_slider_layout.insertWidget(0, self.strength_slider)

        sliders_layout.addLayout(scale_slider_layout)
        sliders_layout.addLayout(strength_slider_layout)
        
        self.dest_label.set_preview_dependencies(self.src_label, self.resize_slider)
        
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(images_layout)
        main_layout.addLayout(sliders_layout)

        self.btn_load_src.clicked.connect(self.load_source)
        self.btn_load_dest.clicked.connect(self.load_destination)
        self.btn_generate.clicked.connect(self.run_cloning)

    def load_source(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Source Image", "", "PNG Images (*.png)")
        if file_path:
            self.src_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if self.src_img.shape[2] != 4: QMessageBox.warning(self, "Warning", "For best results, please use a PNG with transparency.")
            self.display_image(self.src_img, self.src_label); self.dest_label.update()

    def load_destination(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Destination Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path: self.dest_img = cv2.imread(file_path, cv2.IMREAD_COLOR); self.display_image(self.dest_img, self.dest_label)

    # --- NEW: Helper functions for each artistic blend mode ---
    def _blend_normal(self, fg, bg, alpha):
        alpha_norm = cv2.cvtColor(alpha.astype(np.float32) / 255, cv2.COLOR_GRAY2BGR)
        return (alpha_norm * fg + (1 - alpha_norm) * bg).astype(np.uint8)

    def _blend_multiply(self, fg, bg):
        return ((fg.astype(float) * bg.astype(float)) / 255).astype(np.uint8)

    def _blend_screen(self, fg, bg):
        return (255 - (((255 - fg.astype(float)) * (255 - bg.astype(float))) / 255)).astype(np.uint8)

    def _blend_overlay(self, fg, bg):
        fg_float, bg_float = fg.astype(float), bg.astype(float)
        return np.where(bg_float <= 128, (2 * fg_float * bg_float) / 255, 255 - 2 * (255 - fg_float) * (255 - bg_float) / 255).astype(np.uint8)
    
    def _blend_soft_light(self, fg, bg):
        fg_float, bg_float = fg.astype(float) / 255, bg.astype(float) / 255
        return ((1 - 2 * fg_float) * (bg_float ** 2) + 2 * fg_float * bg_float) * 255

    def run_cloning(self):
        center_point, scale = self.dest_label.get_placement_data()
        clone_mode_str = self.clone_method_combo.currentText()
        strength = self.strength_slider.value() / 100.0

        if self.src_img is None or self.dest_img is None or center_point is None:
            QMessageBox.warning(self, "Error", "Please load images and click a destination point."); return

        h, w = self.src_img.shape[:2]
        scaled_src = cv2.resize(self.src_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if scaled_src.shape[2] == 4: mask, src_for_clone = scaled_src[:,:,3], scaled_src[:,:,:3]
        else: mask, src_for_clone = 255 * np.ones(scaled_src.shape[:2], dtype=np.uint8), scaled_src
        
        dest_for_clone = self.dest_img.copy()

        src_h, src_w = src_for_clone.shape[:2]
        top_left_x, top_left_y = center_point[0] - src_w // 2, center_point[1] - src_h // 2
        roi_left, roi_top = max(0, top_left_x), max(0, top_left_y)
        roi_right, roi_bottom = min(dest_for_clone.shape[1], top_left_x + src_w), min(dest_for_clone.shape[0], top_left_y + src_h)

        if roi_right <= roi_left or roi_bottom <= roi_top: return

        src_crop_left, src_crop_top = roi_left - top_left_x, roi_top - top_left_y
        src_crop_right, src_crop_bottom = src_crop_left + (roi_right - roi_left), src_crop_top + (roi_bottom - roi_top)
        
        final_src = src_for_clone[src_crop_top:src_crop_bottom, src_crop_left:src_crop_right]
        final_mask = mask[src_crop_top:src_crop_bottom, src_crop_left:src_crop_right]
        dest_roi = dest_for_clone[roi_top:roi_bottom, roi_left:roi_right]
        
        try:
            if final_src.shape[:2] != dest_roi.shape[:2]: raise ValueError("Shape mismatch")
            
            blended_roi = None
            # --- NEW: Router to select the correct blend function ---
            if clone_mode_str == "Normal (Alpha Blend)":
                blended_roi = self._blend_normal(final_src, dest_roi, final_mask)
            elif clone_mode_str.startswith("Seamless"):
                final_center = (roi_left + (roi_right - roi_left) // 2, roi_top + (roi_bottom - roi_top) // 2)
                mode = cv2.NORMAL_CLONE if clone_mode_str == "Seamless - Normal" else cv2.MIXED_CLONE
                cloned_img = cv2.seamlessClone(final_src, dest_for_clone, final_mask, final_center, mode)
                blended_roi = cloned_img[roi_top:roi_bottom, roi_left:roi_right]
            else: # Artistic Blend Modes
                # First, calculate the 100% strength blend result
                if clone_mode_str == "Multiply (Darken)": blend_result = self._blend_multiply(final_src, dest_roi)
                elif clone_mode_str == "Screen (Lighten)": blend_result = self._blend_screen(final_src, dest_roi)
                elif clone_mode_str == "Overlay (Contrast)": blend_result = self._blend_overlay(final_src, dest_roi)
                elif clone_mode_str == "Soft Light (Subtle)": blend_result = self._blend_soft_light(final_src, dest_roi)
                # Then, use the alpha mask to composite the result onto the original destination
                blended_roi = self._blend_normal(blend_result, dest_roi, final_mask)

            # --- NEW: Universal Strength application ---
            # Blend the result with the original ROI based on the intensity slider
            final_roi = cv2.addWeighted(blended_roi, strength, dest_roi, 1 - strength, 0)

            dest_for_clone[roi_top:roi_bottom, roi_left:roi_right] = final_roi
            self.result_img = dest_for_clone

            self.display_image(self.result_img, self.result_label)
            if QMessageBox.question(self, "Save Image", "Do you want to save the result?") == QMessageBox.StandardButton.Yes:
                self.save_result()
        except (cv2.error, ValueError) as e:
            QMessageBox.critical(self, "Processing Error", f"An error occurred.\nDetails: {e}")

    def save_result(self):
        if self.result_img is None: return
        output_dir = "generation"; os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Result Image", os.path.join(output_dir, f"cloned_{timestamp}.png"), "PNG (*.png);;JPEG (*.jpg)")
        if file_path: cv2.imwrite(file_path, self.result_img); QMessageBox.information(self, "Success", f"Image saved to:\n{file_path}")

    def display_image(self, cv_img, label):
        if cv_img is None: return
        img = cv_img.copy()
        if len(img.shape) == 3 and img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA); fmt = QImage.Format.Format_RGBA8888
        else:
            if len(img.shape) < 3: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format.Format_RGB888
        label.setPixmap(QPixmap.fromImage(QImage(img.data, img.shape[1], img.shape[0], img.strides[0], fmt)))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    cloner_app = SeamlessClonerApp()
    cloner_app.show()
    sys.exit(app.exec())