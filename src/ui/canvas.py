"""Filter-chain canvas (Phase 5, CONTRACTS.md §12, §13).

`FilterChain` (src/python/filters/chain.py) is the single source of truth.
`FilterCanvas` never keeps a private copy of filter parameters -- every
mutating method below calls straight into the model and then rebuilds its
child widgets from `chain.blocks` (CONTRACTS.md §12's ownership rule).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QVBoxLayout, QWidget

from filters import ChainBlock, FilterChain
from ui.palette import MIME_FILTER_KIND
from ui.widgets.filter_block import MIME_BLOCK_ID, FilterBlockWidget


class FilterCanvas(QWidget):
    """Renders `chain.blocks` in order and handles selection/reorder/drop/delete.

    Signals:
        chain_changed: emitted after any model mutation (add/remove/move/clear)
            so the surrounding window can refresh Validate/Export enabled
            state and its dirty indicator.
        selection_changed(object): emitted with the new `selected_id` (a
            block id `str`, or `None` for "Combined") after a selection.
    """

    chain_changed = pyqtSignal()
    selection_changed = pyqtSignal(object)

    def __init__(self, chain: FilterChain, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.chain = chain
        self.setObjectName("filterCanvas")
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(80)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.addStretch(1)

        self._block_widgets: dict[str, FilterBlockWidget] = {}

        # Mutating methods below only touch the model and emit; the view is
        # rebuilt exclusively through these signal/slot connections so
        # "refresh after a model mutation" is never duplicated inline.
        self.chain_changed.connect(self.refresh)
        self.selection_changed.connect(lambda _block_id: self.refresh())
        self.refresh()

    # -- rendering ---------------------------------------------------------

    def refresh(self) -> None:
        """Rebuilds child widgets from `chain.blocks` -- the only place widgets are created."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

        self._block_widgets = {}
        selected = self.chain.selected_id
        for position, block in enumerate(self.chain.blocks):
            widget = FilterBlockWidget(block, position, selected=(block.id == selected), parent=self)
            widget.clicked.connect(self.select_block)
            widget.delete_requested.connect(self.request_remove_block)
            widget.toggle_enabled_requested.connect(self.toggle_block_enabled)
            self._layout.addWidget(widget)
            self._block_widgets[block.id] = widget
        self._layout.addStretch(1)

    # -- mutation (model first, then re-render) -----------------------------

    def add_block(self, kind: str, **params: float) -> str:
        block_id = self.chain.add_block(kind, **params)
        self.chain_changed.emit()
        return block_id

    def remove_block(self, block_id: str) -> None:
        self.chain.remove_block(block_id)
        self.chain_changed.emit()
        self.selection_changed.emit(self.chain.selected_id)

    def request_remove_block(self, block_id: str) -> None:
        """Delete-icon path: confirms before removing (unlike keyboard Delete/Backspace
        in `keyPressEvent`, which calls `remove_block()` directly, unconfirmed)."""
        try:
            block = self.chain.get_block(block_id)
        except KeyError:
            return
        if not self._confirm_delete_block(block):
            return
        self.remove_block(block_id)

    def _confirm_delete_block(self, block: ChainBlock) -> bool:
        reply = QMessageBox.question(
            self,
            "Delete filter?",
            f"Delete this {block.kind} filter block? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def move_block(self, block_id: str, to_index: int) -> None:
        self.chain.move_block(block_id, to_index)
        self.chain_changed.emit()

    def set_block_enabled(self, block_id: str, enabled: bool) -> None:
        self.chain.set_enabled(block_id, enabled)
        self.chain_changed.emit()

    def toggle_block_enabled(self, block_id: str) -> None:
        self.set_block_enabled(block_id, not self.chain.get_block(block_id).enabled)

    def select_block(self, block_id: str | None) -> None:
        self.chain.select(block_id)
        self.selection_changed.emit(block_id)

    def clear(self) -> None:
        """Empties the chain after confirmation (CONTRACTS.md §13). No-op on an empty chain."""
        if not self.chain.blocks:
            return
        if not self._confirm_clear(len(self.chain.blocks)):
            return
        self.chain.clear()
        self.chain_changed.emit()
        self.selection_changed.emit(self.chain.selected_id)

    def _confirm_clear(self, count: int) -> bool:
        reply = QMessageBox.question(
            self,
            "Clear all filters?",
            f"Clear all {count} filter block{'s' if count != 1 else ''}? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    # -- keyboard ------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            selected = self.chain.selected_id
            if selected is not None:
                self.remove_block(selected)
                return
        super().keyPressEvent(event)

    # -- drag and drop ---------------------------------------------------------

    def _index_for_y(self, y: int) -> int:
        """Chain index the drop position corresponds to, based on child midpoints."""
        for index, block in enumerate(self.chain.blocks):
            widget = self._block_widgets.get(block.id)
            if widget is not None and y < widget.geometry().center().y():
                return index
        return len(self.chain.blocks)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        mime = event.mimeData()
        if mime.hasFormat(MIME_FILTER_KIND) or mime.hasFormat(MIME_BLOCK_ID):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        mime = event.mimeData()
        pos = event.position().toPoint()
        if mime.hasFormat(MIME_BLOCK_ID):
            block_id = bytes(mime.data(MIME_BLOCK_ID)).decode("utf-8")
            self.move_block(block_id, self._index_for_y(pos.y()))
            event.acceptProposedAction()
        elif mime.hasFormat(MIME_FILTER_KIND):
            kind = bytes(mime.data(MIME_FILTER_KIND)).decode("utf-8")
            self.add_block(kind)
            event.acceptProposedAction()
        else:
            event.ignore()
