"""Signal-block canvas for the Time-Domain view (docs/CONCEPT.md §11.7).

`SignalChain` (src/python/signals/chain.py) is the single source of truth,
mirroring `ui/canvas.py`'s `FilterCanvas`/`FilterChain` ownership rule:
`SignalCanvas` never keeps a private copy of block parameters -- every
mutating method calls into the model first and then rebuilds tiles from
`signal_chain.blocks`.

Blocks are shown as a flat, unordered list feeding a sum (§11.2/§11.7) --
there are no series connectors and no reordering, since summation order
doesn't matter (unlike the filter chain's series canvas). They are stacked
vertically (one per row) rather than side by side, since the list can grow
long and a single scrollable column reads better than horizontal scrolling.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QMessageBox, QVBoxLayout, QWidget

from signals import SignalChain, SignalChainBlock
from ui.signal_palette import MIME_SIGNAL_KIND
from ui.widgets.signal_block import SignalBlockWidget


class SignalCanvas(QWidget):
    """Renders `signal_chain.blocks` and handles drop/edit/delete.

    Signals:
        chain_changed: emitted after any model mutation (add/remove/edit/
            clear) so the surrounding view can refresh its Inspector plots.
    """

    chain_changed = pyqtSignal()

    def __init__(self, signal_chain: SignalChain, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.signal_chain = signal_chain
        self.setObjectName("signalCanvas")
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)

        self._placeholder = QLabel("Drag signal blocks here (summed — order doesn't matter)")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888;")
        self._layout.addWidget(self._placeholder)

        self._blocks_column = QVBoxLayout()
        self._layout.addLayout(self._blocks_column)
        self._layout.addStretch(1)

        self.chain_changed.connect(self.refresh)
        self.refresh()

    # -- rendering -----------------------------------------------------------

    def refresh(self) -> None:
        """Rebuilds child widgets from `signal_chain.blocks` -- the only place widgets are created."""
        while self._blocks_column.count():
            item = self._blocks_column.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        blocks = self.signal_chain.blocks
        self._placeholder.setVisible(not blocks)

        for block in blocks:
            tile = SignalBlockWidget(block, parent=self)
            tile.params_edited.connect(self.update_params)
            tile.delete_requested.connect(self.request_remove_block)
            tile.reseed_requested.connect(self.reseed_block)
            self._blocks_column.addWidget(tile, 0, Qt.AlignmentFlag.AlignLeft)

        if blocks:
            sigma = QLabel("Σ")
            sigma.setStyleSheet("font-size: 18pt;")
            self._blocks_column.addWidget(sigma, 0, Qt.AlignmentFlag.AlignHCenter)
        self._blocks_column.addStretch(1)

    # -- mutation (model first, then re-render via chain_changed) -----------------

    def add_block(self, kind: str) -> str:
        block_id = self.signal_chain.add_block(kind)
        self.chain_changed.emit()
        return block_id

    def update_params(self, block_id: str, params: dict) -> None:
        params = dict(params)
        if "factor" in params:
            self.signal_chain.set_factor(block_id, params.pop("factor"))
        if params:
            self.signal_chain.update_params(block_id, **params)
        self.chain_changed.emit()

    def remove_block(self, block_id: str) -> None:
        self.signal_chain.remove_block(block_id)
        self.chain_changed.emit()

    def reseed_block(self, block_id: str) -> None:
        try:
            self.signal_chain.reseed(block_id)
        except KeyError:
            return
        self.chain_changed.emit()

    def request_remove_block(self, block_id: str) -> None:
        try:
            block = self.signal_chain.get_block(block_id)
        except KeyError:
            return
        if not self._confirm_delete_block(block):
            return
        self.remove_block(block_id)

    def _confirm_delete_block(self, block: SignalChainBlock) -> bool:
        reply = QMessageBox.question(
            self,
            "Delete signal block?",
            f"Delete this {block.kind} signal block? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def clear(self) -> None:
        if not self.signal_chain.blocks:
            return
        self.signal_chain.clear()
        self.chain_changed.emit()

    # -- drag and drop ---------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.mimeData().hasFormat(MIME_SIGNAL_KIND):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        mime = event.mimeData()
        if not mime.hasFormat(MIME_SIGNAL_KIND):
            return
        kind = bytes(mime.data(MIME_SIGNAL_KIND)).decode("utf-8")
        self.add_block(kind)
        event.acceptProposedAction()
