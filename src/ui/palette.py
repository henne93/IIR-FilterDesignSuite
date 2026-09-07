"""Draggable filter-kind palette (Phase 5, CONTRACTS.md §14 phase-5 row).

Each entry is a drag *source* only -- it carries no filter parameters of its
own (CONTRACTS.md §12 ownership rule: no widget maintains independent filter
state). Dropping an entry onto the canvas asks `FilterCanvas` to
`chain.add_block(kind)`, which applies `filters.chain.DEFAULT_PARAMS`.
"""

from __future__ import annotations

from PyQt6.QtCore import QMimeData, QPoint, Qt
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget

from filters import BlockKind

MIME_FILTER_KIND = "application/x-iir-filter-kind"

PALETTE_KINDS: tuple[tuple[BlockKind, str], ...] = (
    ("LP", "Low-Pass"),
    ("HP", "High-Pass"),
    ("BP", "Band-Pass"),
    ("AP", "All-Pass"),
    ("PK", "Peak"),
)


class PaletteEntry(QFrame):
    """One draggable palette tile for a single filter kind."""

    def __init__(self, kind: BlockKind, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("paletteEntry")
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
        mime.setData(MIME_FILTER_KIND, self.kind.encode("utf-8"))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)
        self._drag_start = None


class FilterPalette(QWidget):
    """Vertical stack of the four draggable filter-kind entries."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("filterPalette")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        title = QLabel("Filters")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        for kind, label in PALETTE_KINDS:
            layout.addWidget(PaletteEntry(kind, label))
        layout.addStretch(1)
