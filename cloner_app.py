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
from PyQt6.QtCore import Qt, QPoint, QRect, QPointF

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
        self._pixmap = pixmap; self.update()

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
        if self._pixmap.isNull(): super().paintEvent(event); return
        target_rect = self._get_target_rect()
        painter = QPainter(self)
        painter.drawPixmap(target_rect, self._pixmap)

class ClickableImageLabel(ScalableImageLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proportional_center, self.source_label, self.resize_slider, self.rotate_slider = None, None, None, None

    def set_preview_dependencies(self, source_label, resize_slider, rotate_slider):
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
        if self.proportional_center and self.source_label and not self.source_label.pixmap().isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            target_rect = self._get_target_rect()
            center_x, center_y = target_rect.x() + self.proportional_center.x() * target_rect.width(), target_rect.y() + self.proportional_center.y() * target_rect.height()
            source_scale, source_pixmap = self.resize_slider.value() / 100.0, self.source_label.pixmap()
            dest_scale_ratio = target_rect.width() / self._pixmap.width()
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
        scale, angle = self.resize_slider.value() / 100.0, self.rotate_slider.value()
        return (center_x, center_y), scale, angle

class SeamlessClonerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Seamless Cloner"); self.setGeometry(100, 100, 1200, 800)
        self.src_img, self.dest_img, self.result_img = None, None, None
        self.initUI()

    def initUI(self):
        central_widget, main_layout = QWidget(), QVBoxLayout()
        self.setCentralWidget(central_widget); main_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout, images_layout, sliders_layout = QHBoxLayout(), QHBoxLayout(), QVBoxLayout()

        self.btn_load_src = QPushButton("1. Load Source (PNG)")
        self.btn_load_dest = QPushButton("2. Load Destination")
        self.clone_method_combo = QComboBox(); self.clone_method_combo.addItems(["Alpha Blend (Strong)", "Seamless - Normal", "Seamless - Mixed"])
        self.btn_generate = QPushButton("5. Generate")
        for w in [self.btn_load_src, self.btn_load_dest, self.clone_method_combo, self.btn_generate]: controls_layout.addWidget(w)
        controls_layout.insertWidget(2, QLabel("3. Blend Method:"))

        self.src_label, self.dest_label, self.result_label = ScalableImageLabel("Source"), ClickableImageLabel("Destination"), ScalableImageLabel("Result")
        for w in [self.src_label, self.dest_label, self.result_label]: images_layout.addWidget(w)

        controls_title, scale_slider_layout, rotate_slider_layout, strength_slider_layout = QLabel("<b>4. Controls:</b>"), QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        self.resize_slider, self.rotate_slider, self.strength_slider = QSlider(Qt.Orientation.Horizontal), QSlider(Qt.Orientation.Horizontal), QSlider(Qt.Orientation.Horizontal)
        self.scale_label, self.rotate_label, self.strength_label = QLabel(), QLabel(), QLabel()
        
        for slider, label, r_min, r_max, r_def in [(self.resize_slider, self.scale_label, 10, 200, 100), (self.rotate_slider, self.rotate_label, -180, 180, 0), (self.strength_slider, self.strength_label, 0, 100, 100)]:
            slider.setRange(r_min, r_max); slider.setValue(r_def)
        
        self.resize_slider.valueChanged.connect(lambda v: self.scale_label.setText(f"{v}%"))
        self.rotate_slider.valueChanged.connect(lambda v: self.rotate_label.setText(f"{v}°"))
        self.strength_slider.valueChanged.connect(lambda v: self.strength_label.setText(f"{v}%"))
        self.scale_label.setText(f"{self.resize_slider.value()}%"); self.rotate_label.setText(f"{self.rotate_slider.value()}°"); self.strength_label.setText(f"{self.strength_slider.value()}%")

        for layout, label_text, slider, text_label in [(scale_slider_layout, "Scale:", self.resize_slider, self.scale_label), (rotate_slider_layout, "Rotate:", self.rotate_slider, self.rotate_label), (strength_slider_layout, "Intensity:", self.strength_slider, self.strength_label)]:
            layout.addWidget(QLabel(label_text)); layout.addWidget(slider); layout.addWidget(text_label)

        for layout in [scale_slider_layout, rotate_slider_layout, strength_slider_layout]: sliders_layout.addLayout(layout)
        sliders_layout.insertWidget(0, controls_title)
        
        self.dest_label.set_preview_dependencies(self.src_label, self.resize_slider, self.rotate_slider)
        for layout in [controls_layout, images_layout, sliders_layout]: main_layout.addLayout(layout)
        central_widget.setLayout(main_layout)
        self.btn_load_src.clicked.connect(self.load_source); self.btn_load_dest.clicked.connect(self.load_destination); self.btn_generate.clicked.connect(self.run_cloning)

    def load_source(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Source Image", "", "PNG Images (*.png)")
        if file_path:
            self.src_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if len(self.src_img.shape) < 3 or self.src_img.shape[2] != 4: QMessageBox.warning(self, "Warning", "For best results, please use a PNG with a transparency layer.")
            self.display_image(self.src_img, self.src_label); self.dest_label.update()

    def load_destination(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Destination Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path: self.dest_img = cv2.imread(file_path, cv2.IMREAD_COLOR); self.display_image(self.dest_img, self.dest_label)

    def _rotate_image_and_mask(self, image, mask, angle):
        (h, w) = image.shape[:2]; (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - cX; M[1, 2] += (new_h / 2) - cY
        rotated_image = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0,0,0,0))
        rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h))
        return rotated_image, rotated_mask

    def run_cloning(self):
        center_point, scale, angle = self.dest_label.get_placement_data()
        clone_mode_str = self.clone_method_combo.currentText()
        strength = self.strength_slider.value() / 100.0

        if self.src_img is None or self.dest_img is None or center_point is None:
            QMessageBox.warning(self, "Error", "Please load images and click a destination point."); return

        try:
            h, w = self.src_img.shape[:2]
            scaled_src = cv2.resize(self.src_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            if len(scaled_src.shape) < 3 or scaled_src.shape[2] != 4:
                mask, src_bgra = 255*np.ones(scaled_src.shape[:2], dtype=np.uint8), cv2.cvtColor(scaled_src, cv2.COLOR_BGR2BGRA)
            else:
                mask, src_bgra = scaled_src[:,:,3], scaled_src
            
            rotated_src_bgra, final_mask = self._rotate_image_and_mask(src_bgra, mask, -angle)
            final_src_bgr = rotated_src_bgra[:,:,:3]
            
            dest_h, dest_w = self.dest_img.shape[:2]
            src_h, src_w = final_src_bgr.shape[:2]
            
            top_left_x, top_left_y = center_point[0] - src_w // 2, center_point[1] - src_h // 2
            bottom_right_x, bottom_right_y = top_left_x + src_w, top_left_y + src_h
            
            # --- FINAL FIX: Detect if out-of-bounds and switch mode if necessary ---
            is_out_of_bounds = (top_left_x < 0) or (top_left_y < 0) or (bottom_right_x > dest_w) or (bottom_right_y > dest_h)
            
            if is_out_of_bounds and clone_mode_str.startswith("Seamless"):
                QMessageBox.information(self, "Mode Switched", "Seamless Clone is not supported for objects placed off-screen.\nSwitching to 'Alpha Blend' for this generation.")
                clone_mode_str = "Alpha Blend (Strong)"

            # --- Unified Clipping and Pasting Logic ---
            dest_roi_x1, dest_roi_y1 = max(0, top_left_x), max(0, top_left_y)
            dest_roi_x2, dest_roi_y2 = min(dest_w, bottom_right_x), min(dest_h, bottom_right_y)
            
            src_crop_x1, src_crop_y1 = dest_roi_x1 - top_left_x, dest_roi_y1 - top_left_y
            src_crop_x2, src_crop_y2 = src_crop_x1 + (dest_roi_x2 - dest_roi_x1), src_crop_y1 + (dest_roi_y2 - dest_roi_y1)
            
            if (dest_roi_x2 - dest_roi_x1) <= 0 or (dest_roi_y2 - dest_roi_y1) <= 0:
                 QMessageBox.warning(self, "Error", "Source image is completely outside the destination."); return

            clipped_src = final_src_bgr[src_crop_y1:src_crop_y2, src_crop_x1:src_crop_x2]
            clipped_mask = final_mask[src_crop_y1:src_crop_y2, src_crop_x1:src_crop_x2]
            
            if clone_mode_str == "Alpha Blend (Strong)":
                dest_roi = self.dest_img[dest_roi_y1:dest_roi_y2, dest_roi_x1:dest_roi_x2]
                alpha_3ch = cv2.cvtColor((clipped_mask.astype(np.float32) * strength) / 255, cv2.COLOR_GRAY2BGR)
                fg, bg = clipped_src.astype(np.float32), dest_roi.astype(np.float32)
                blended_roi = (cv2.multiply(alpha_3ch, fg) + cv2.multiply(1.0 - alpha_3ch, bg)).astype(np.uint8)
                self.result_img = self.dest_img.copy()
                self.result_img[dest_roi_y1:dest_roi_y2, dest_roi_x1:dest_roi_x2] = blended_roi
            else: # Seamless Modes (now only runs when fully in-bounds)
                dest_for_clone = self.dest_img.copy()
                mode = cv2.NORMAL_CLONE if clone_mode_str == "Seamless - Normal" else cv2.MIXED_CLONE
                cloned_at_100 = cv2.seamlessClone(clipped_src, dest_for_clone, clipped_mask, center_point, mode)
                self.result_img = cv2.addWeighted(cloned_at_100, strength, self.dest_img, 1 - strength, 0)

            self.display_image(self.result_img, self.result_label)
            if QMessageBox.question(self, "Save Image", "Do you want to save the result?") == QMessageBox.StandardButton.Yes: self.save_result()
        except (cv2.error, ValueError) as e:
            QMessageBox.critical(self, "Processing Error", f"An error occurred.\n\nDetails: {e}")

    def save_result(self):
        if self.result_img is None: return
        output_dir = "generation"; os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Result", os.path.join(output_dir, f"cloned_{timestamp}.png"), "PNG (*.png);;JPEG (*.jpg)")
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
    app = QApplication(sys.argv); cloner_app = SeamlessClonerApp(); cloner_app.show(); sys.exit(app.exec())