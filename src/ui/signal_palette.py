"""Draggable signal-block palette for the Time-Domain view (docs/CONCEPT.md §11.7).

Rough GUI pass: mirrors `ui/palette.py`'s drag-source pattern for the
filter-kind palette, but for the signal-generator building blocks (Sine, DC,
Noise, CSV Import) that will later be summed into the time-domain source
signal (§11.2). No signal model exists yet -- dropping an entry currently
just adds a placeholder block to `SignalCanvas`.
"""

from __future__ import annotations

from PyQt6.QtCore import QMimeData, QPoint, Qt
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget

MIME_SIGNAL_KIND = "application/x-iir-signal-kind"

SIGNAL_KINDS: tuple[tuple[str, str], ...] = (
    ("SIN", "Sine"),
    ("DC", "DC"),
    ("NOISE", "Noise"),
    ("CSV", "CSV Import"),
)


class SignalPaletteEntry(QFrame):
    """One draggable palette tile for a single signal-block kind."""

    def __init__(self, kind: str, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("signalPaletteEntry")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(f"Drag onto the canvas to add a {label} block")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        kind_label = QLabel(kind)
        kind_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(kind_label)
        layout.addWidget(QLabel(label))

        self._drag_start: QPoint | None = None

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not (event.buttons() & Qt.MouseButton.LeftButton) or self._drag_start is None:
            return
        moved = event.position().toPoint() - self._drag_start
        if moved.manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(MIME_SIGNAL_KIND, self.kind.encode("utf-8"))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)
        self._drag_start = None


class SignalPalette(QWidget):
    """Vertical stack of the draggable signal-block-kind entries."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("signalPalette")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        title = QLabel("Signal blocks")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        for kind, label in SIGNAL_KINDS:
            layout.addWidget(SignalPaletteEntry(kind, label))
        layout.addStretch(1)
