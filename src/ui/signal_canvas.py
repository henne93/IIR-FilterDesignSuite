"""Signal-block canvas for the Time-Domain view (docs/CONCEPT.md §11.7).

Rough GUI pass only: there is no `SignalChain` model yet (the time-domain
equivalent of `filters.chain.FilterChain`), so this widget keeps its own
throwaway list of dropped block kinds purely for visual placeholder feedback
-- no parameters, no `factor` field, no actual signal generation. A future
iteration replaces `_kinds` with a real model, once one exists, the same way
`ui/canvas.py` renders from `FilterChain`.

Blocks are shown as a flat, unordered list feeding a sum (docs/CONCEPT.md
§11.2/§11.7) -- there are no series connectors here, unlike the filter
chain's `FilterCanvas`, since summation order doesn't matter.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ui.signal_palette import MIME_SIGNAL_KIND


class SignalCanvas(QWidget):
    """Placeholder canvas: shows dropped signal-block kinds summed into one source signal."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("signalCanvas")
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)

        self._kinds: list[str] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)

        self._placeholder = QLabel("Drag signal blocks here (summed — order doesn't matter)")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888;")
        self._layout.addWidget(self._placeholder)

        self._blocks_row = QHBoxLayout()
        self._layout.addLayout(self._blocks_row)
        self._layout.addStretch(1)

    # -- drag and drop -------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.mimeData().hasFormat(MIME_SIGNAL_KIND):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        mime = event.mimeData()
        if not mime.hasFormat(MIME_SIGNAL_KIND):
            return
        kind = bytes(mime.data(MIME_SIGNAL_KIND)).decode("utf-8")
        self._kinds.append(kind)
        self.refresh()
        event.acceptProposedAction()

    # -- rendering -----------------------------------------------------------

    def refresh(self) -> None:
        while self._blocks_row.count():
            item = self._blocks_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self._placeholder.setVisible(not self._kinds)

        for kind in self._kinds:
            tile = QFrame()
            tile.setFrameShape(QFrame.Shape.StyledPanel)
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(8, 6, 8, 6)
            kind_label = QLabel(kind)
            kind_label.setStyleSheet("font-weight: bold;")
            tile_layout.addWidget(kind_label)
            self._blocks_row.addWidget(tile)

        if self._kinds:
            sigma = QLabel("Σ")
            sigma.setStyleSheet("font-size: 18pt;")
            self._blocks_row.addWidget(sigma)
        self._blocks_row.addStretch(1)

    def clear(self) -> None:
        self._kinds = []
        self.refresh()
