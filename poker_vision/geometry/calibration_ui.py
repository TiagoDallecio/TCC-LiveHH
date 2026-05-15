"""
Calibration UI — PyQt6 tool for placing fiducials on a table frame and
producing a homography profile.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from poker_vision.geometry.calibration_profile import CalibrationProfile, FiducialEntry
from poker_vision.geometry.calibrator import TableCalibrator

# ---------------------------------------------------------------------------
# Constants — canonical fiducial layout (hardcoded for v1; YAML-bound later)
# ---------------------------------------------------------------------------

CANONICAL_W, CANONICAL_H = 1000, 600

# ---------------------------------------------------------------------------
# Constants — canonical fiducial layout (B2C POV)
# ---------------------------------------------------------------------------

# Mapeamento: (nome_interno, (X_canonico, Y_canonico), label_para_o_usuario)
FIDUCIALS: list[tuple[str, tuple[float, float], str]] = [
    ("pov_top_left", (200.0, 0.0), "① Fundo Esquerdo (Lado do Pote)"),
    ("pov_top_right", (800.0, 0.0), "② Fundo Direito (Lado do Pote)"),
    ("pov_bottom_right", (800.0, 600.0), "③ Base Direita (Perto do Celular)"),
    ("pov_bottom_left", (200.0, 600.0), "④ Base Esquerda (Perto do Celular)"),
]

LOUPE_SIZE_PX = 160  # size of the loupe widget on screen
LOUPE_ZOOM = 5  # magnification factor
LOUPE_SAMPLE_PX = LOUPE_SIZE_PX // LOUPE_ZOOM  # source region size on the frame


@dataclass
class Marker:
    name: str
    label: str
    image_xy: QPointF  # in source frame coordinates (not widget!)
    is_outlier: bool = False


class ImageView(QLabel):
    """
    Displays the current video frame, captures clicks, renders markers,
    and draws a zoom loupe near the cursor.
    """

    clicked = pyqtSignal(QPointF)  # emits frame-space coordinates
    right_clicked = pyqtSignal(QPointF)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #1e1e1e;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._frame: Optional[np.ndarray] = None  # BGR ndarray
        self._pixmap: Optional[QPixmap] = None
        self.markers: dict[str, Marker] = {}
        self._cursor_widget_pos: Optional[QPoint] = None
        self._show_loupe = False

    # ---- frame management ----

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        self._frame = frame_bgr.copy()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)  # type: ignore[call-overload]
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self.update()

    def frame_size(self) -> tuple[int, int]:
        if self._frame is None:
            return (0, 0)
        return self._frame.shape[1], self._frame.shape[0]

    # ---- coordinate mapping ----

    def _displayed_rect(self) -> QRectF:
        """The rectangle (in widget coords) where the scaled pixmap is drawn."""
        if self._pixmap is None:
            return QRectF()
        w_widget, h_widget = self.width(), self.height()
        w_pix, h_pix = self._pixmap.width(), self._pixmap.height()
        scale = min(w_widget / w_pix, h_widget / h_pix)
        draw_w, draw_h = w_pix * scale, h_pix * scale
        x = (w_widget - draw_w) / 2
        y = (h_widget - draw_h) / 2
        return QRectF(x, y, draw_w, draw_h)

    def _widget_to_frame(self, pt: QPoint | QPointF) -> Optional[QPointF]:
        if self._pixmap is None:
            return None
        rect = self._displayed_rect()
        if not rect.contains(QPointF(pt)):
            return None
        fx = (pt.x() - rect.x()) / rect.width() * self._pixmap.width()
        fy = (pt.y() - rect.y()) / rect.height() * self._pixmap.height()
        return QPointF(fx, fy)

    def _frame_to_widget(self, pt: QPointF) -> QPointF:
        rect = self._displayed_rect()
        assert self._pixmap is not None
        x = rect.x() + pt.x() / self._pixmap.width() * rect.width()
        y = rect.y() + pt.y() / self._pixmap.height() * rect.height()
        return QPointF(x, y)

    # ---- mouse events ----

    def mousePressEvent(self, ev: Optional[QMouseEvent]) -> None:
        if ev is None:
            return
        frame_pt = self._widget_to_frame(ev.position())
        if frame_pt is None:
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(frame_pt)
        elif ev.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(frame_pt)

    def mouseMoveEvent(self, ev: Optional[QMouseEvent]) -> None:
        if ev is None:
            return
        self._cursor_widget_pos = ev.position().toPoint()
        self._show_loupe = self._widget_to_frame(ev.position()) is not None
        self.update()

    def leaveEvent(self, _: object) -> None:
        self._show_loupe = False
        self.update()

    # ---- painting ----

    def paintEvent(self, _) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        if self._pixmap is None:
            painter.setPen(QColor("#888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame loaded — open a video file.")
            return

        rect = self._displayed_rect()
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))

        # Markers
        for marker in self.markers.values():
            self._draw_marker(painter, marker)

        # Loupe
        if self._show_loupe and self._cursor_widget_pos is not None:
            self._draw_loupe(painter, self._cursor_widget_pos)

    def _draw_marker(self, painter: QPainter, marker: Marker) -> None:
        widget_pt = self._frame_to_widget(marker.image_xy)
        color = QColor("#ff5050") if marker.is_outlier else QColor("#50ff80")
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 90)))
        painter.drawEllipse(widget_pt, 8, 8)
        painter.setPen(QPen(QColor("white"), 1))
        font = QFont()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(widget_pt + QPointF(12, -8), marker.label.split(" ")[0])

    def _draw_loupe(self, painter: QPainter, cursor_widget_pt: QPoint) -> None:
        assert self._frame is not None
        frame_pt = self._widget_to_frame(cursor_widget_pt)
        if frame_pt is None:
            return

        # Source rectangle on the frame
        half = LOUPE_SAMPLE_PX // 2
        cx, cy = int(frame_pt.x()), int(frame_pt.y())
        h, w = self._frame.shape[:2]
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, cx + half)
        y1 = min(h, cy + half)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return

        crop = self._frame[y0:y1, x0:x1]
        zoomed = cv2.resize(crop, (LOUPE_SIZE_PX, LOUPE_SIZE_PX), interpolation=cv2.INTER_NEAREST)
        # Crosshair
        mid = LOUPE_SIZE_PX // 2
        cv2.line(zoomed, (mid, 0), (mid, LOUPE_SIZE_PX), (0, 255, 255), 1)
        cv2.line(zoomed, (0, mid), (LOUPE_SIZE_PX, mid), (0, 255, 255), 1)

        rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, LOUPE_SIZE_PX, LOUPE_SIZE_PX, LOUPE_SIZE_PX * 3, QImage.Format.Format_RGB888).copy()  # type: ignore[call-overload]

        # Position the loupe near the cursor but inside the widget
        offset_x, offset_y = 24, 24
        lx = cursor_widget_pt.x() + offset_x
        ly = cursor_widget_pt.y() + offset_y
        if lx + LOUPE_SIZE_PX > self.width():
            lx = cursor_widget_pt.x() - LOUPE_SIZE_PX - offset_x
        if ly + LOUPE_SIZE_PX > self.height():
            ly = cursor_widget_pt.y() - LOUPE_SIZE_PX - offset_y

        target = QRect(int(lx), int(ly), LOUPE_SIZE_PX, LOUPE_SIZE_PX)
        painter.drawImage(target, qimg)
        painter.setPen(QPen(QColor("#ffff00"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(target)

    # ---- marker API ----

    def add_or_update_marker(self, name: str, label: str, image_xy: QPointF) -> None:
        self.markers[name] = Marker(name=name, label=label, image_xy=image_xy)
        self.update()

    def remove_marker(self, name: str) -> None:
        self.markers.pop(name, None)
        self.update()

    def nearest_marker(self, image_xy: QPointF, max_dist_px: float = 20.0) -> Optional[str]:
        best, best_d = None, max_dist_px
        for name, m in self.markers.items():
            d = float(np.hypot(m.image_xy.x() - image_xy.x(), m.image_xy.y() - image_xy.y()))
            if d < best_d:
                best, best_d = name, d
        return best

    def set_outliers(self, outlier_names: set[str]) -> None:
        for name, m in self.markers.items():
            m.is_outlier = name in outlier_names
        self.update()


# ---------------------------------------------------------------------------
# Warp preview dialog
# ---------------------------------------------------------------------------


class WarpPreviewDialog(QDialog):
    def __init__(self, warped_bgr: np.ndarray, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Warp Preview — canonical top-down view")
        rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()  # type: ignore[call-overload]
        label = QLabel()
        label.setPixmap(QPixmap.fromImage(qimg))
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        self.resize(w + 24, h + 24)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class CalibrationWindow(QMainWindow):
    def __init__(self, video_path: Optional[Path] = None) -> None:
        super().__init__()
        self.setWindowTitle("Poker Vision — Table Calibrator")
        self.resize(1400, 800)

        self.calibrator = TableCalibrator()
        self.video_path: Optional[Path] = None
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.current_frame_index: int = 0
        self.current_frame: Optional[np.ndarray] = None

        self.current_fiducial_idx: int = 0  # pointer into FIDUCIALS list

        self._build_ui()
        self._wire_shortcuts()

        if video_path is not None:
            self._load_video(video_path)

    # ---- UI construction ----

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Top toolbar
        toolbar = QHBoxLayout()
        self.open_btn = QPushButton("Open Video…")
        self.load_profile_btn = QPushButton("Load Profile…")
        self.preview_btn = QPushButton("Preview Warp")
        self.save_btn = QPushButton("Save Profile…")
        self.reset_btn = QPushButton("Reset")
        for b in (self.open_btn, self.load_profile_btn, self.preview_btn, self.save_btn, self.reset_btn):
            toolbar.addWidget(b)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        # Body — split into image view + right panel
        body = QHBoxLayout()
        self.image_view = ImageView()
        body.addWidget(self.image_view, stretch=4)

        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Fiducials to place</b>"))
        self.fid_list = QListWidget()
        self.fid_list.setMinimumWidth(280)
        for name, _, label in FIDUCIALS:
            item = QListWidgetItem(f"○  {label}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.fid_list.addItem(item)
        right.addWidget(self.fid_list)

        self.error_label = QLabel("Reprojection error: —")
        self.error_label.setStyleSheet("font-weight: bold; padding: 6px;")
        right.addWidget(self.error_label)

        right.addStretch(1)
        body.addLayout(right, stretch=1)
        root.addLayout(body, stretch=1)

        # Video seek slider
        seek_row = QHBoxLayout()
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_label = QLabel("Frame: — / —")
        seek_row.addWidget(QLabel("Seek:"))
        seek_row.addWidget(self.seek_slider, stretch=1)
        seek_row.addWidget(self.seek_label)
        root.addLayout(seek_row)

        # Status bar
        self.setStatusBar(QStatusBar())
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage("Open a video to begin.")

        # Signals
        self.open_btn.clicked.connect(self._on_open_clicked)
        self.load_profile_btn.clicked.connect(self._on_load_profile)
        self.preview_btn.clicked.connect(self._on_preview_warp)
        self.save_btn.clicked.connect(self._on_save_profile)
        self.reset_btn.clicked.connect(self._on_reset)
        self.fid_list.currentRowChanged.connect(self._on_fiducial_selected)
        self.image_view.clicked.connect(self._on_image_clicked)
        self.image_view.right_clicked.connect(self._on_image_right_clicked)
        self.seek_slider.valueChanged.connect(self._on_seek)

        self._refresh_active_fiducial()

    def _wire_shortcuts(self) -> None:
        shortcut_tab = QShortcut(QKeySequence("Tab"), self)
        shortcut_tab.activated.connect(self._advance_to_next_pending)

        shortcut_del = QShortcut(QKeySequence(Qt.Key.Key_Backspace), self)
        shortcut_del.activated.connect(self._delete_last)

        shortcut_space = QShortcut(QKeySequence("Space"), self)
        shortcut_space.activated.connect(self._on_preview_warp)

    # ---- video handling ----

    def _on_open_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video files (*.mp4 *.mov *.mkv *.avi)")
        if path_str:
            self._load_video(Path(path_str))

    def _load_video(self, path: Path) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video: {path}")
            return
        self.video_cap = cap
        self.video_path = path
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.seek_slider.setEnabled(True)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(max(0, self.frame_count - 1))
        self.seek_slider.setValue(0)
        self._seek_to(0)
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(f"Loaded {path.name} — {self.frame_count} frames")

    def _on_seek(self, value: int) -> None:
        self._seek_to(value)

    def _seek_to(self, idx: int) -> None:
        if self.video_cap is None:
            return
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.video_cap.read()
        if not ok or frame is None:
            return
        self.current_frame = frame
        self.current_frame_index = idx
        self.image_view.set_frame(frame)
        self.seek_label.setText(f"Frame: {idx} / {self.frame_count - 1}")

    # ---- fiducial workflow ----

    def _refresh_active_fiducial(self) -> None:
        # Find first not-yet-placed fiducial, or keep current if all done
        placed = set(self.image_view.markers.keys())
        pending_idx = next(
            (i for i, (n, _, _) in enumerate(FIDUCIALS) if n not in placed),
            None,
        )
        target = pending_idx if pending_idx is not None else self.current_fiducial_idx
        self.current_fiducial_idx = target
        self.fid_list.setCurrentRow(target)
        self._update_list_labels()

    def _update_list_labels(self) -> None:
        for i in range(self.fid_list.count()):
            name, _, label = FIDUCIALS[i]
            placed = name in self.image_view.markers
            prefix = "✓ " if placed else ("➤ " if i == self.current_fiducial_idx else "○ ")
            item = self.fid_list.item(i)
            if item is not None:
                item.setText(f"{prefix} {label}")

    def _on_fiducial_selected(self, row: int) -> None:
        if 0 <= row < len(FIDUCIALS):
            self.current_fiducial_idx = row
            self._update_list_labels()

    def _advance_to_next_pending(self) -> None:
        placed = set(self.image_view.markers.keys())
        for i in range(len(FIDUCIALS)):
            idx = (self.current_fiducial_idx + 1 + i) % len(FIDUCIALS)
            if FIDUCIALS[idx][0] not in placed:
                self.current_fiducial_idx = idx
                self.fid_list.setCurrentRow(idx)
                self._update_list_labels()
                return

    def _delete_last(self) -> None:
        if not self.image_view.markers:
            return
        last_name = list(self.image_view.markers.keys())[-1]
        self.image_view.remove_marker(last_name)
        self._after_markers_changed()

    # ---- click handlers ----

    def _on_image_clicked(self, frame_pt: QPointF) -> None:
        if self.current_frame is None:
            return
        name, _, label = FIDUCIALS[self.current_fiducial_idx]
        self.image_view.add_or_update_marker(name, label, frame_pt)
        self._advance_to_next_pending()
        self._after_markers_changed()

    def _on_image_right_clicked(self, frame_pt: QPointF) -> None:
        name = self.image_view.nearest_marker(frame_pt)
        if name is not None:
            self.image_view.remove_marker(name)
            self._after_markers_changed()

    def _after_markers_changed(self) -> None:
        self._recompute_homography()
        self._update_list_labels()

    # ---- homography ----

    def _recompute_homography(self) -> None:
        markers = self.image_view.markers
        if len(markers) < 4:
            self.error_label.setText("Reprojection error: — (need ≥4 fiducials)")
            self.error_label.setStyleSheet("color: #888; font-weight: bold; padding: 6px;")
            self.image_view.set_outliers(set())
            return

        canonical_map = {name: pt for name, pt, _ in FIDUCIALS}
        image_points = {n: (m.image_xy.x(), m.image_xy.y()) for n, m in markers.items()}
        canonical_points = {n: canonical_map[n] for n in image_points}

        ok = self.calibrator.calibrate_from_fiducials(image_points, canonical_points)
        if not ok:
            self.error_label.setText("Reprojection error: — (homography failed)")
            self.error_label.setStyleSheet("color: #ff5050; font-weight: bold; padding: 6px;")
            return

        err = self.calibrator.median_error
        if err < 3.0:
            color = "#50ff80"  # green
            grade = "GOOD"
        elif err < 8.0:
            color = "#ffcc00"  # yellow
            grade = "OK"
        else:
            color = "#ff5050"  # red
            grade = "POOR"

        self.error_label.setText(f"Reprojection error: {err:.2f} px — {grade}")
        self.error_label.setStyleSheet(f"color: {color}; font-weight: bold; padding: 6px;")

        # Outlier flagging if the calibrator exposes an inlier mask
        outlier_names: set[str] = set()
        mask = getattr(self.calibrator, "inlier_mask", None)
        if mask is not None:
            sorted_names = sorted(image_points.keys())
            for i, n in enumerate(sorted_names):
                if not bool(mask.ravel()[i]):
                    outlier_names.add(n)
        self.image_view.set_outliers(outlier_names)

    # ---- preview / save / load / reset ----

    def _on_preview_warp(self) -> None:
        if self.calibrator.H is None or self.current_frame is None:
            QMessageBox.information(self, "Preview", "Place at least 4 fiducials first.")
            return
        warped = self.calibrator.warp_frame(self.current_frame, (CANONICAL_W, CANONICAL_H))
        WarpPreviewDialog(warped, self).exec()

    def _on_save_profile(self) -> None:
        if self.calibrator.H is None:
            QMessageBox.warning(self, "Save", "Cannot save: homography not computed yet.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save calibration profile", "calibration_profile.yaml", "YAML files (*.yaml *.yml)"
        )
        if not path_str:
            return

        canonical_map = dict((n, c) for n, c, _ in FIDUCIALS)
        entries = [
            FiducialEntry(
                name=n,
                canonical=canonical_map[n],
                image=(m.image_xy.x(), m.image_xy.y()),
            )
            for n, m in self.image_view.markers.items()
        ]
        profile = CalibrationProfile(
            profile_id=Path(path_str).stem,
            created_at=datetime.now(timezone.utc).isoformat(),
            canonical_size=(CANONICAL_W, CANONICAL_H),
            fiducials=entries,
            homography=self.calibrator.H.tolist(),
            reprojection_error_median_px=float(self.calibrator.median_error),
            source_video=str(self.video_path) if self.video_path else None,
            source_frame_index=self.current_frame_index,
        )
        profile.save(Path(path_str))
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(f"Saved profile to {path_str}", 5000)

    def _on_load_profile(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Load calibration profile", "", "YAML files (*.yaml *.yml)")
        if not path_str:
            return
        try:
            profile = CalibrationProfile.load(Path(path_str))
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return

        self.image_view.markers.clear()
        label_map = {n: lbl for n, _, lbl in FIDUCIALS}
        for entry in profile.fiducials:
            if entry.name in label_map:
                self.image_view.add_or_update_marker(
                    entry.name,
                    label_map[entry.name],
                    QPointF(entry.image[0], entry.image[1]),
                )
        self._after_markers_changed()
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(f"Loaded profile from {path_str}", 5000)

    def _on_reset(self) -> None:
        confirm = QMessageBox.question(self, "Reset", "Clear all placed fiducials?")
        if confirm == QMessageBox.StandardButton.Yes:
            self.image_view.markers.clear()
            self.calibrator = TableCalibrator()
            self.current_fiducial_idx = 0
            self.fid_list.setCurrentRow(0)
            self._after_markers_changed()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run_calibration_ui(video_path: Optional[Path] = None) -> int:
    app = QApplication(sys.argv)
    win = CalibrationWindow(video_path=video_path)
    win.show()
    return app.exec()


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(run_calibration_ui(arg))
