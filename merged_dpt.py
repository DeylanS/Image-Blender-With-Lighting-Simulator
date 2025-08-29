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
    """
    QLabel that scales its pixmap with aspect ratio. Supports an optional
    'light direction' indicator controlled by a QSlider (angle).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet("QLabel { color: grey; }")

        # Optional: a QSlider controlling an angle for drawing indicator
        self._light_angle_slider = None

    def set_light_indicator_slider(self, angle_slider: QSlider | None):
        """Provide a QSlider to draw a yellow 'light direction' indicator."""
        self._light_angle_slider = angle_slider
        if self._light_angle_slider:
            self._light_angle_slider.valueChanged.connect(self.update)

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
        painter = QPainter(self)

        if self._pixmap.isNull():
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
            return

        # Draw scaled pixmap
        target_rect = self._get_target_rect()
        painter.drawPixmap(target_rect, self._pixmap)

        # Draw light direction indicator if slider provided
        if self._light_angle_slider is not None:
            angle = self._light_angle_slider.value()
            center = target_rect.center()
            length = int(min(target_rect.width(), target_rect.height()) * 0.4)

            # Angle adjusted by -180 to indicate "light coming from"
            rad = np.deg2rad(angle - 180)
            end_x = center.x() + length * np.cos(rad)
            end_y = center.y() + length * np.sin(rad)

            pen = QPen(QColor(255, 255, 0, 220), 2, Qt.PenStyle.DashLine)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(pen)
            painter.drawLine(center.x(), center.y(), int(end_x), int(end_y))
            painter.drawEllipse(center, 3, 3)


class ClickableImageLabel(ScalableImageLabel):
    """
    Destination label that accepts clicks to set placement and shows a live
    overlay preview of the source (scaled/rotated) with a dashed bounding box.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proportional_center = None
        self.source_label = None
        self.resize_slider = None
        self.rotate_slider = None

    def set_preview_dependencies(self, source_label, resize_slider: QSlider, rotate_slider: QSlider):
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
                    relative_y / target_rect.height(),
                )
                self.update()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        if (
            self.proportional_center
            and self.source_label
            and self.source_label.pixmap()
            and not self.source_label.pixmap().isNull()
            and self.resize_slider
            and self.rotate_slider
        ):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            target_rect = self._get_target_rect()

            center_x = target_rect.x() + self.proportional_center.x() * target_rect.width()
            center_y = target_rect.y() + self.proportional_center.y() * target_rect.height()

            source_scale = self.resize_slider.value() / 100.0
            source_pixmap = self.source_label.pixmap()

            # Adjust preview scale to match label scaling
            dest_scale_ratio = (
                target_rect.width() / self._pixmap.width()
                if self._pixmap.width() > 0
                else 0
            )
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
        dest_w, dest_h = self._pixmap.width(), self._pixmap.height()
        center_x = int(self.proportional_center.x() * dest_w)
        center_y = int(self.proportional_center.y() * dest_h)
        scale = self.resize_slider.value() / 100.0 if self.resize_slider else 1.0
        angle = self.rotate_slider.value() if self.rotate_slider else 0
        return (center_x, center_y), scale, angle


# ================
#      Main
# ================
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Compositor")
        self.setGeometry(100, 100, 1400, 850)

        self.src_img_original = None
        self.src_img_lit = None
        self.dest_img = None
        self.result_img = None

        # Default light color (BGR)
        self.light_color = (255, 255, 220)

        self.initUI()

    # ---------------
    # UI Construction
    # ---------------
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top bar
        top_layout = QHBoxLayout()
        self.btn_load_src = QPushButton("Load Source")
        self.btn_load_dest = QPushButton("Load Destination")
        self.cloner_method_combo = QComboBox()
        self.cloner_method_combo.addItems(
            ["Alpha Blend / Feather", "Seamless - Normal", "Seamless - Mixed"]
        )
        self.btn_generate = QPushButton("Generate")

        top_layout.addWidget(self.btn_load_src)
        top_layout.addWidget(self.btn_load_dest)
        top_layout.addWidget(QLabel("Blend Method:"))
        top_layout.addWidget(self.cloner_method_combo)
        top_layout.addWidget(self.btn_generate)
        main_layout.addLayout(top_layout)

        # Images splitter (fills remaining height)
        images_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.src_label = ScalableImageLabel("1. Load Source")
        self.dest_label = ClickableImageLabel("2. Load Destination")
        self.result_label = ScalableImageLabel("Result")

        images_splitter.addWidget(self.src_label)
        images_splitter.addWidget(self.dest_label)
        images_splitter.addWidget(self.result_label)
        images_splitter.setSizes([350, 500, 500])
        main_layout.addWidget(images_splitter, 1)

        # Bottom tabs (full width)
        self.tabs = QTabWidget()
        self.tabs.setFixedHeight(240)
        main_layout.addWidget(self.tabs)  # full width

        # ---- Cloning Tab ----
        cloning_tab = QWidget()
        cloning_form = QFormLayout(cloning_tab)

        self.lighting_checkbox = QCheckBox("Enable Source Lighting")
        cloning_form.addRow(self.lighting_checkbox)

        # Slider rows for cloning
        self.resize_slider_w = self._create_slider_with_lineedit(10, 300, 100)   # percent
        self.rotate_slider_w = self._create_slider_with_lineedit(-180, 180, 0)  # degrees
        self.strength_slider_w = self._create_slider_with_lineedit(0, 100, 100) # blend %
        self.feather_slider_w = self._create_slider_with_lineedit(0, 50, 0)     # px-ish

        cloning_form.addRow("Scale (%):", self.resize_slider_w)
        cloning_form.addRow("Rotate (°):", self.rotate_slider_w)
        cloning_form.addRow("Intensity (%):", self.strength_slider_w)
        cloning_form.addRow("Feather:", self.feather_slider_w)

        self.tabs.addTab(cloning_tab, "Cloning Controls")

        # ---- Lighting Tab ----
        lighting_tab = QWidget()
        lighting_form = QFormLayout(lighting_tab)

        self.btn_color = QPushButton("Choose Light Color")
        lighting_form.addRow(self.btn_color)

        self.light_blend_combo = QComboBox()
        self.light_blend_combo.addItems(
            ["Overlay", "Soft Light", "Screen (Lighten)", "Multiply (Darken)"]
        )
        lighting_form.addRow("Light Blend:", self.light_blend_combo)

        self.angle_slider_w = self._create_slider_with_lineedit(0, 360, 45)
        self.intensity_slider_w = self._create_slider_with_lineedit(0, 100, 75)
        self.smoothness_slider_w = self._create_slider_with_lineedit(10, 300, 100)

        lighting_form.addRow("Light Angle (°):", self.angle_slider_w)
        lighting_form.addRow("Light Intensity (%):", self.intensity_slider_w)
        lighting_form.addRow("Gradient Smoothness:", self.smoothness_slider_w)

        self.tabs.addTab(lighting_tab, "Source Lighting")
        self.tabs.setTabEnabled(1, False)  # disabled until checkbox is checked

        # Link the yellow direction indicator to the source label
        self.src_label.set_light_indicator_slider(self.angle_slider_w.findChild(QSlider))

        # ---- Connections ----
        self.lighting_checkbox.toggled.connect(self.toggle_lighting_tab)

        # destination preview overlay depends on scale & rotate
        self.dest_label.set_preview_dependencies(
            self.src_label,
            self.resize_slider_w.findChild(QSlider),
            self.rotate_slider_w.findChild(QSlider),
        )

        self.btn_load_src.clicked.connect(self.load_source)
        self.btn_load_dest.clicked.connect(self.load_destination)
        self.btn_generate.clicked.connect(self.run_cloning)

        self.btn_color.clicked.connect(self.choose_color)
        self.light_blend_combo.currentIndexChanged.connect(self._apply_lighting_to_source)
        for w in [self.angle_slider_w, self.intensity_slider_w, self.smoothness_slider_w]:
            w.findChild(QSlider).valueChanged.connect(self._apply_lighting_to_source)

    # -------------------------
    #      UI Helper Pieces
    # -------------------------
    def _create_slider_with_lineedit(self, min_val, max_val, default_val):
        """
        Returns a QWidget containing a horizontal QSlider and a QLineEdit that
        are two-way synced. The slider expands to fill row width.
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        line_edit = QLineEdit(str(default_val))
        line_edit.setValidator(QIntValidator(min_val, max_val, self))
        line_edit.setFixedWidth(55)

        slider.valueChanged.connect(lambda v: line_edit.setText(str(v)))
        line_edit.editingFinished.connect(lambda: slider.setValue(int(line_edit.text())))

        layout.addWidget(slider)
        layout.addWidget(line_edit)
        return widget

    # -------------------------
    #      Event Handlers
    # -------------------------
    def toggle_lighting_tab(self, checked):
        self.tabs.setTabEnabled(1, checked)
        self._apply_lighting_to_source()

    def load_source(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Source Image", "", "PNG Images (*.png);;All Images (*.png *.jpg *.jpeg)"
        )
        if not file_path:
            return

        self.src_img_original = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if self.src_img_original is None:
            QMessageBox.critical(self, "Error", "Failed to read source image.")
            return

        if len(self.src_img_original.shape) < 3 or (
            self.src_img_original.shape[2] != 4
        ):
            QMessageBox.warning(
                self,
                "Warning",
                "For best results, use a PNG with a transparency layer.",
            )

        # Immediately show, respecting current lighting toggle
        self._apply_lighting_to_source()

    def load_destination(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Destination Image",
            "",
            "Image Files (*.png *.jpg *.jpeg)",
        )
        if not file_path:
            return

        self.dest_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if self.dest_img is None:
            QMessageBox.critical(self, "Error", "Failed to read destination image.")
            return
        self.display_image(self.dest_img, self.dest_label)

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            # store as BGR tuple
            self.light_color = (color.blue(), color.green(), color.red())
            self._apply_lighting_to_source()
            # style button for feedback
            self.btn_color.setStyleSheet(
                f"background-color: {color.name()}; "
                f"color: {'black' if color.lightness() > 127 else 'white'};"
            )

    # -------------------------
    #      Lighting Pipeline
    # -------------------------
    def _apply_lighting_to_source(self):
        """
        Applies the lighting effect to the current source image and updates the
        Source preview. If lighting is disabled or no source image: no-op.
        """
        if self.src_img_original is None:
            return

        if not self.lighting_checkbox.isChecked():
            # lighting disabled -> show original
            self.src_img_lit = self.src_img_original.copy()
            self.display_image(self.src_img_lit, self.src_label)
            self.src_label.update()  # update the angle line
            self.dest_label.update()
            return

        # Split alpha if present
        if len(self.src_img_original.shape) == 3 and self.src_img_original.shape[2] == 4:
            has_alpha = True
            original_alpha = self.src_img_original[:, :, 3]
            base_image_bgr = self.src_img_original[:, :, :3]
        else:
            has_alpha = False
            base_image_bgr = (
                cv2.cvtColor(self.src_img_original, cv2.COLOR_GRAY2BGR)
                if len(self.src_img_original.shape) < 3
                else self.src_img_original
            )

        angle = self.angle_slider_w.findChild(QSlider).value()
        intensity = self.intensity_slider_w.findChild(QSlider).value()
        smoothness = self.smoothness_slider_w.findChild(QSlider).value()
        blend_mode = self.light_blend_combo.currentText()

        h, w = base_image_bgr.shape[:2]
        gradient_bgra = self._create_gradient(w, h, angle, self.light_color, intensity, smoothness)
        gradient_bgr = gradient_bgra[:, :, :3]
        gradient_alpha = gradient_bgra[:, :, 3]

        blended_base = self._blend(gradient_bgr, base_image_bgr, blend_mode)

        alpha_norm = cv2.cvtColor(gradient_alpha.astype(np.float32) / 255.0, cv2.COLOR_GRAY2BGR)
        processed_bgr = (
            base_image_bgr.astype(np.float32) * (1 - alpha_norm) + blended_base * alpha_norm
        ).astype(np.uint8)

        if has_alpha:
            self.src_img_lit = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2BGRA)
            self.src_img_lit[:, :, 3] = original_alpha
        else:
            self.src_img_lit = processed_bgr

        self.display_image(self.src_img_lit, self.src_label)
        self.src_label.update()  # redraw indicator
        self.dest_label.update()  # refresh overlay

    def _create_gradient(self, w, h, angle, color_bgr, intensity, smoothness):
        """
        Creates a BGRA gradient image with alpha modulated by intensity & smoothness.
        """
        rad = np.deg2rad(angle)
        vx, vy = np.cos(rad), np.sin(rad)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        projection = x * vx + y * vy
        min_proj, max_proj = projection.min(), projection.max()

        if max_proj == min_proj:
            gradient_norm = np.zeros((h, w), dtype=np.float32)
        else:
            gradient_norm = (projection - min_proj) / (max_proj - min_proj)

        # Smoothness remap (0.2 .. 4.0) with your earlier mapping
        exponent = 0.2 + (smoothness - 10) / (300 - 10) * (4.0 - 0.2)
        alpha_channel = (gradient_norm ** exponent * (intensity / 100.0) * 255).astype(np.uint8)

        color_layer = np.full((h, w, 3), color_bgr, dtype=np.uint8)
        return cv2.merge(
            [color_layer[:, :, 0], color_layer[:, :, 1], color_layer[:, :, 2], alpha_channel]
        )

    def _blend(self, fg, bg, mode):
        """
        Porter-Duff style blend modes between two BGR images (uint8).
        """
        fg_float = fg.astype(np.float32)
        bg_float = bg.astype(np.float32)

        if mode == "Multiply (Darken)":
            result = (fg_float * bg_float / 255.0)
        elif mode == "Screen (Lighten)":
            result = 255.0 - ((255.0 - fg_float) * (255.0 - bg_float) / 255.0)
        elif mode == "Overlay":
            result = np.where(
                bg_float <= 128.0,
                (2.0 * fg_float * bg_float) / 255.0,
                255.0 - 2.0 * (255.0 - fg_float) * (255.0 - bg_float) / 255.0,
            )
        elif mode == "Soft Light":
            a = fg_float / 255.0
            b = bg_float / 255.0
            result = ((1 - 2 * a) * (b ** 2) + 2 * a * b) * 255.0
        else:
            result = bg_float

        return np.clip(result, 0, 255).astype(np.uint8)

    # -------------------------
    #      Cloning Pipeline
    # -------------------------
    def run_cloning(self):
        center_point, scale, angle = self.dest_label.get_placement_data()
        clone_mode_str = self.cloner_method_combo.currentText()
        strength = self.strength_slider_w.findChild(QSlider).value() / 100.0
        feather_amount = self.feather_slider_w.findChild(QSlider).value()

        if self.src_img_lit is None or self.dest_img is None or center_point is None:
            QMessageBox.warning(self, "Error", "Please load images and click a destination point.")
            return

        try:
            # scale source
            src_h0, src_w0 = self.src_img_lit.shape[:2]
            new_w = max(1, int(round(src_w0 * scale)))
            new_h = max(1, int(round(src_h0 * scale)))
            scaled_src = cv2.resize(self.src_img_lit, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # extract mask
            if len(scaled_src.shape) < 3 or scaled_src.shape[2] != 4:
                src_bgra = cv2.cvtColor(scaled_src, cv2.COLOR_BGR2BGRA)
                mask = np.full(scaled_src.shape[:2], 255, dtype=np.uint8)
            else:
                src_bgra = scaled_src
                mask = scaled_src[:, :, 3]

            # rotate
            rotated_src_bgra, final_mask = self._rotate_image_and_mask(src_bgra, mask, -angle)
            final_src_bgr = rotated_src_bgra[:, :, :3]

            # placement bbox
            dest_h, dest_w = self.dest_img.shape[:2]
            src_h, src_w = final_src_bgr.shape[:2]
            top_left_x = center_point[0] - src_w // 2
            top_left_y = center_point[1] - src_h // 2
            bottom_right_x = top_left_x + src_w
            bottom_right_y = top_left_y + src_h

            # seamless clone needs in-bounds (non-edge)
            is_oob = (
                (top_left_x < 1)
                or (top_left_y < 1)
                or (bottom_right_x >= dest_w - 1)
                or (bottom_right_y >= dest_h - 1)
            )
            if is_oob and clone_mode_str.startswith("Seamless"):
                QMessageBox.information(
                    self,
                    "Mode Switched",
                    "Seamless Clone requires a border and fails at the image edge.\n"
                    "Switching to 'Alpha Blend' for this generation.",
                )
                clone_mode_str = "Alpha Blend / Feather"

            # clip to ROI
            dest_roi_x1 = max(0, top_left_x)
            dest_roi_y1 = max(0, top_left_y)
            dest_roi_x2 = min(dest_w, bottom_right_x)
            dest_roi_y2 = min(dest_h, bottom_right_y)

            src_crop_x1 = dest_roi_x1 - top_left_x
            src_crop_y1 = dest_roi_y1 - top_left_y
            src_crop_x2 = src_crop_x1 + (dest_roi_x2 - dest_roi_x1)
            src_crop_y2 = src_crop_y1 + (dest_roi_y2 - dest_roi_y1)

            if (dest_roi_x2 - dest_roi_x1) <= 0 or (dest_roi_y2 - dest_roi_y1) <= 0:
                return

            clipped_src = final_src_bgr[src_crop_y1:src_crop_y2, src_crop_x1:src_crop_x2]
            clipped_mask = final_mask[src_crop_y1:src_crop_y2, src_crop_x1:src_crop_x2]

            if clone_mode_str.startswith("Alpha Blend"):
                if feather_amount > 0:
                    k_size = max(1, feather_amount * 2 + 1)
                    clipped_mask = cv2.GaussianBlur(clipped_mask, (k_size, k_size), 0)

                dest_roi = self.dest_img[dest_roi_y1:dest_roi_y2, dest_roi_x1:dest_roi_x2]
                alpha_3ch = cv2.cvtColor(
                    (clipped_mask.astype(np.float32) * strength) / 255.0,
                    cv2.COLOR_GRAY2BGR,
                )
                fg = clipped_src.astype(np.float32)
                bg = dest_roi.astype(np.float32)
                blended_roi = (cv2.multiply(alpha_3ch, fg) + cv2.multiply(1.0 - alpha_3ch, bg)).astype(
                    np.uint8
                )
                self.result_img = self.dest_img.copy()
                self.result_img[dest_roi_y1:dest_roi_y2, dest_roi_x1:dest_roi_x2] = blended_roi
            else:
                mode = cv2.NORMAL_CLONE if clone_mode_str == "Seamless - Normal" else cv2.MIXED_CLONE
                safe_center = (
                    dest_roi_x1 + (dest_roi_x2 - dest_roi_x1) // 2,
                    dest_roi_y1 + (dest_roi_y2 - dest_roi_y1) // 2,
                )
                cloned_at_100 = cv2.seamlessClone(
                    clipped_src, self.dest_img.copy(), clipped_mask, safe_center, mode
                )
                self.result_img = cv2.addWeighted(cloned_at_100, strength, self.dest_img, 1 - strength, 0)

            self.display_image(self.result_img, self.result_label)

            if (
                QMessageBox.question(self, "Save Image", "Do you want to save the result?")
                == QMessageBox.StandardButton.Yes
            ):
                self.save_result()

        except (cv2.error, ValueError) as e:
            QMessageBox.critical(self, "Processing Error", f"An error occurred.\n\nDetails: {e}")

    def _rotate_image_and_mask(self, image_bgra, mask, angle_deg):
        (h, w) = image_bgra.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - cX
        M[1, 2] += (new_h / 2) - cY

        rotated_image = cv2.warpAffine(
            image_bgra, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)
        )
        rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
        return rotated_image, rotated_mask

    # -------------------------
    #         Saving
    # -------------------------
    def save_result(self):
        if self.result_img is None:
            return
        output_dir = "generation"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            os.path.join(output_dir, f"composited_{timestamp}.png"),
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if file_path:
            cv2.imwrite(file_path, self.result_img)
            QMessageBox.information(self, "Success", f"Image saved to:\n{file_path}")

    # -------------------------
    #       Display Utils
    # -------------------------
    def display_image(self, img_cv, label: QLabel):
        if img_cv is None:
            return
        img = img_cv.copy()
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            h, w = img_rgba.shape[:2]
            qimg = QImage(img_rgba.data, w, h, img_rgba.strides[0], QImage.Format.Format_RGBA8888)
        else:
            if len(img.shape) < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            qimg = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))


# ============
#   Entry
# ============
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainApp()
    ex.show()
    sys.exit(app.exec())
