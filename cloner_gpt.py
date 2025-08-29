import sys
import os
import cv2
import math
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
        if self._pixmap.isNull():
            return QRect()
        label_size = self.size()
        scaled_pixmap = self._pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
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
            dest_scale_ratio = target_rect.width() / self._pixmap.width()
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

class SeamlessClonerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Seamless Cloner")
        self.setGeometry(100, 100, 1200, 800)
        self.src_img = None
        self.dest_img = None
        self.result_img = None
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        self.setCentralWidget(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        controls_layout = QHBoxLayout()
        images_layout = QHBoxLayout()
        sliders_layout = QVBoxLayout()

        self.btn_load_src = QPushButton("1. Load Source (PNG)")
        self.btn_load_dest = QPushButton("2. Load Destination")
        self.clone_method_combo = QComboBox()
        self.clone_method_combo.addItems(["Alpha Blend (Strong)", "Seamless - Normal", "Seamless - Mixed"])
        self.btn_generate = QPushButton("5. Generate")

        for w in [self.btn_load_src, self.btn_load_dest, self.clone_method_combo, self.btn_generate]:
            controls_layout.addWidget(w)
        controls_layout.insertWidget(2, QLabel("3. Blend Method:"))

        self.src_label = ScalableImageLabel("Source")
        self.dest_label = ClickableImageLabel("Destination")
        self.result_label = ScalableImageLabel("Result")
        for w in [self.src_label, self.dest_label, self.result_label]:
            images_layout.addWidget(w)

        controls_title = QLabel("<b>4. Controls:</b>")
        scale_slider_layout = QHBoxLayout()
        rotate_slider_layout = QHBoxLayout()
        strength_slider_layout = QHBoxLayout()

        self.resize_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotate_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_label = QLabel()
        self.rotate_label = QLabel()
        self.strength_label = QLabel()

        for slider, label, r_min, r_max, r_def in [
            (self.resize_slider, self.scale_label, 10, 200, 100),
            (self.rotate_slider, self.rotate_label, -180, 180, 0),
            (self.strength_slider, self.strength_label, 0, 100, 100),
        ]:
            slider.setRange(r_min, r_max)
            slider.setValue(r_def)

        self.resize_slider.valueChanged.connect(lambda v: self.scale_label.setText(f"{v}%"))
        self.rotate_slider.valueChanged.connect(lambda v: self.rotate_label.setText(f"{v}°"))
        self.strength_slider.valueChanged.connect(lambda v: self.strength_label.setText(f"{v}%"))

        self.scale_label.setText(f"{self.resize_slider.value()}%")
        self.rotate_label.setText(f"{self.rotate_slider.value()}°")
        self.strength_label.setText(f"{self.strength_slider.value()}%")

        for layout, label_text, slider, text_label in [
            (scale_slider_layout, "Scale:", self.resize_slider, self.scale_label),
            (rotate_slider_layout, "Rotate:", self.rotate_slider, self.rotate_label),
            (strength_slider_layout, "Intensity:", self.strength_slider, self.strength_label),
        ]:
            layout.addWidget(QLabel(label_text))
            layout.addWidget(slider)
            layout.addWidget(text_label)

        for layout in [scale_slider_layout, rotate_slider_layout, strength_slider_layout]:
            sliders_layout.addLayout(layout)
        sliders_layout.insertWidget(0, controls_title)

        self.dest_label.set_preview_dependencies(self.src_label, self.resize_slider, self.rotate_slider)

        for layout in [controls_layout, images_layout, sliders_layout]:
            main_layout.addLayout(layout)
        central_widget.setLayout(main_layout)

        self.btn_load_src.clicked.connect(self.load_source)
        self.btn_load_dest.clicked.connect(self.load_destination)
        self.btn_generate.clicked.connect(self.run_cloning)

    def load_source(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Source Image", "", "PNG Images (*.png)")
        if file_path:
            self.src_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if self.src_img is None:
                QMessageBox.critical(self, "Error", "Failed to read source image.")
                return
            if self.src_img.shape[2] != 4:
                QMessageBox.warning(self, "Warning", "For best results, use a PNG with transparency.")
            self.display_image(self.src_img, self.src_label)
            self.dest_label.update()

    def load_destination(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Destination Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.dest_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if self.dest_img is None:
                QMessageBox.critical(self, "Error", "Failed to read destination image.")
                return
            self.display_image(self.dest_img, self.dest_label)

    # --- REMOVED _rotate_image_and_mask helper. Logic is now simpler and inside run_cloning ---

    def run_cloning(self):
        center_point, scale, angle = self.dest_label.get_placement_data()
        clone_mode_str = self.clone_method_combo.currentText()
        strength = self.strength_slider.value() / 100.0

        if self.src_img is None or self.dest_img is None or center_point is None:
            QMessageBox.warning(self, "Error", "Please load images and click a destination point.")
            return

        try:
            # 1) Scale source image (guard against 0-sized)
            h, w = self.src_img.shape[:2]
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            scaled_src = cv2.resize(self.src_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 2) Ensure BGRA before rotation
            if scaled_src.shape[2] != 4:
                scaled_src = cv2.cvtColor(scaled_src, cv2.COLOR_BGR2BGRA)

            dest_h, dest_w = self.dest_img.shape[:2]

            # 3) Rotation centered on source image (negative to match preview direction)
            src_h, src_w = scaled_src.shape[:2]
            rot_mat = cv2.getRotationMatrix2D((src_w / 2, src_h / 2), -angle, 1.0)

            # 4) Translate to destination click location
            rot_mat[0, 2] += center_point[0] - src_w / 2
            rot_mat[1, 2] += center_point[1] - src_h / 2

            # 5) Warp onto full-size transparent canvas (handles out-of-bounds gracefully)
            full_canvas_bgra = cv2.warpAffine(
                scaled_src, rot_mat, (dest_w, dest_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
            )

            # 6) Extract BGR source and 8-bit mask
            final_mask = full_canvas_bgra[:, :, 3]
            final_src_canvas = full_canvas_bgra[:, :, :3]

            if clone_mode_str == "Alpha Blend (Strong)":
                # Manual alpha blend with intensity
                alpha = (final_mask.astype(np.float32) * strength) / 255.0
                alpha_3ch = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
                fg = final_src_canvas.astype(np.float32)
                bg = self.dest_img.astype(np.float32)
                blended = cv2.multiply(alpha_3ch, fg) + cv2.multiply(1.0 - alpha_3ch, bg)
                self.result_img = blended.astype(np.uint8)

            else:
                # --- Seamless modes with robust cropping to avoid assertion ---
                mode = cv2.NORMAL_CLONE if clone_mode_str == "Seamless - Normal" else cv2.MIXED_CLONE

                # a) Find nonzero region of mask
                ys, xs = np.where(final_mask > 0)
                if len(xs) == 0 or len(ys) == 0:
                    QMessageBox.warning(self, "Error", "Source mask is empty after transformation.")
                    return

                # b) Bounding box of visible (in-bounds) pixels
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()

                # c) Clip ROI to destination boundaries
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(dest_w - 1, x_max)
                y_max = min(dest_h - 1, y_max)

                if x_max < x_min or y_max < y_min:
                    QMessageBox.warning(self, "Error", "Nothing left inside image bounds to blend.")
                    return

                roi_src = final_src_canvas[y_min:y_max + 1, x_min:x_max + 1]
                roi_mask = final_mask[y_min:y_max + 1, x_min:x_max + 1]

                if roi_src.size == 0 or roi_mask.size == 0:
                    QMessageBox.warning(self, "Error", "Nothing left inside image bounds to blend.")
                    return

                # d) Binary mask (required by seamlessClone)
                _, roi_mask_bin = cv2.threshold(roi_mask, 0, 255, cv2.THRESH_BINARY)
                roi_mask_bin = roi_mask_bin.astype(np.uint8)

                # e) Center in destination coords (must be integers)
                cropped_center = (int(round(center_point[0])), int(round(center_point[1])))

                # f) Perform seamlessClone
                cloned_at_100 = cv2.seamlessClone(
                    roi_src,                # src (ROI, 3ch)
                    self.dest_img,          # dst (full image, 3ch)
                    roi_mask_bin,           # mask (ROI, 1ch, 0/255)
                    cropped_center,         # center in destination coordinates
                    mode
                )

                # g) Blend with intensity slider (0..1)
                self.result_img = cv2.addWeighted(cloned_at_100, strength, self.dest_img, 1 - strength, 0)

            # 7) Display and optional save
            self.display_image(self.result_img, self.result_label)
            if QMessageBox.question(self, "Save Image", "Do you want to save the result?") == QMessageBox.StandardButton.Yes:
                self.save_result()

        except (cv2.error, ValueError) as e:
            QMessageBox.critical(self, "Processing Error", f"An error occurred.\n\nDetails: {e}")

    def save_result(self):
        if self.result_img is None:
            return
        output_dir = "generation"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            os.path.join(output_dir, f"cloned_{timestamp}.png"),
            "PNG (*.png);;JPEG (*.jpg)"
        )
        if file_path:
            cv2.imwrite(file_path, self.result_img)
            QMessageBox.information(self, "Success", f"Image saved to:\n{file_path}")

    def display_image(self, cv_img, label):
        if cv_img is None:
            return
        img = cv_img.copy()
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            fmt = QImage.Format.Format_RGBA8888
        else:
            if len(img.shape) < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format.Format_RGB888
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], fmt)
        label.setPixmap(QPixmap.fromImage(qimg))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    cloner_app = SeamlessClonerApp()
    cloner_app.show()
    sys.exit(app.exec())
