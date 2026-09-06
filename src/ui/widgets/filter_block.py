"""Single filter-block widget rendered on the canvas (Phase 5).

Ownership rule (CONTRACTS.md §12): this widget holds only a `block_id` and
renders whatever `FilterChain.get_block(block_id)` currently reports -- it
never stores or computes filter parameters itself. `FilterCanvas` rebuilds
these widgets from the model on every `refresh()`; nothing here mutates the
model directly except by emitting signals the canvas listens to.
"""

from __future__ import annotations

from PyQt6.QtCore import QMimeData, QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from filters import ChainBlock

MIME_BLOCK_ID = "application/x-iir-block-id"

_KIND_NAMES = {"LP": "Low-Pass", "HP": "High-Pass", "BP": "Band-Pass", "AP": "All-Pass"}

# Inline action-control sizing (module docstring: buttons live *in* the
# block's own header row -- fixed height as well as width so a QToolButton's
# native size hint can never push the row taller than the title/param text
# that shares it).
_ACTION_BUTTON_SIZE = QSize(20, 20)

# Toggle icon: same glyph for both states (a filled/open circle reads as an
# on/off indicator without relying on an external icon asset) -- only the
# color changes, per the enabled=green / disabled=grey convention.
_TOGGLE_ENABLED_GLYPH = "●"  # "●" -- active
_TOGGLE_DISABLED_GLYPH = "○"  # "○" -- bypassed
_TOGGLE_ENABLED_COLOR = "#27ae60"  # green
_TOGGLE_DISABLED_COLOR = "#999999"  # grey (CONTRACTS-adjacent: red is reserved for invalid/error text)


def _format_params(block: ChainBlock) -> str:
    """Human-readable parameter summary sourced entirely from the model.

    No filter math is reimplemented here -- `block.params` is exactly what
    was passed to `FilterChain.add_block`/`update_params`, and
    `block.filter.derived_params()` (when valid) is the same dict
    `BandPassFilter`/etc. already compute (CONTRACTS.md §1).
    """
    parts = [f"{name}={value:g} Hz" if "f" in name.lower() else f"{name}={value:g}" for name, value in block.params.items()]
    if block.filter is not None:
        derived = block.filter.derived_params()
        parts += [f"{name}={value:g}" for name, value in derived.items()]
    return ", ".join(parts)


class FilterBlockWidget(QFrame):
    """Renders one `ChainBlock`: kind, params, validity, enabled/selection state.

    The delete and enable/disable buttons never touch `ChainBlock`/
    `FilterChain` directly (ownership rule, module docstring) -- they only
    emit `delete_requested`/`toggle_enabled_requested`; `FilterCanvas` is the
    one that calls into the model and decides whether deletion needs
    confirming.
    """

    clicked = pyqtSignal(str)  # block_id
    delete_requested = pyqtSignal(str)  # block_id
    toggle_enabled_requested = pyqtSignal(str)  # block_id

    def __init__(self, block: ChainBlock, position: int, selected: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.block_id = block.id
        self.setObjectName("filterBlock")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(2)

        kind_name = _KIND_NAMES.get(block.kind, block.kind)
        title = f"FILT{position + 1}: {kind_name} ({block.kind})"
        if not block.enabled:
            title += " (disabled)"
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold;")
        header.addWidget(self.title_label, 1)

        self._position = position

        # Both action controls are constructed with `self` as their parent
        # up front (rather than relying on a later `layout.addLayout(header)`
        # to reparent them) and go straight into `header`, which is itself
        # installed into this widget's own `layout` below -- there is no
        # point at which either button exists without this frame as its
        # parent, so neither can ever surface as an independent top-level
        # window. `setAutoRaise(True)` plus a fixed 20x20 size keeps them
        # flat, compact inline controls rather than bordered mini-dialogs.
        self.toggle_button = QToolButton(self)
        self.toggle_button.setObjectName("toggleEnabledButton")
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setFixedSize(_ACTION_BUTTON_SIZE)
        self.toggle_button.clicked.connect(lambda: self.toggle_enabled_requested.emit(self.block_id))
        header.addWidget(self.toggle_button)

        self.delete_button = QToolButton(self)
        self.delete_button.setObjectName("deleteBlockButton")
        self.delete_button.setAutoRaise(True)
        self.delete_button.setFixedSize(_ACTION_BUTTON_SIZE)
        self.delete_button.setText("✕")
        self.delete_button.setToolTip("Delete this filter block")
        self.delete_button.setAccessibleName(f"Delete FILT{position + 1}")
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self.block_id))
        header.addWidget(self.delete_button)

        layout.addLayout(header)

        self.params_label = QLabel(_format_params(block))
        layout.addWidget(self.params_label)

        self.error_label = QLabel(block.error or "")
        self.error_label.setObjectName("blockError")
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(not block.is_valid)
        layout.addWidget(self.error_label)

        self.set_selected(selected)
        self.set_valid(block.is_valid)
        self.set_enabled_state(block.enabled)

        self._drag_start: QPoint | None = None

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        border = "2px solid #2980b9" if selected else "1px solid #999999"
        self.setStyleSheet(f"#filterBlock {{ border: {border}; border-radius: 4px; }}")

    def set_valid(self, valid: bool) -> None:
        self.valid = valid
        self.error_label.setVisible(not valid)

    def set_enabled_state(self, enabled: bool) -> None:
        """Bypass indicator (`ChainBlock.enabled`) -- purely visual, never mutates the model."""
        self.enabled_state = enabled
        dim = "" if enabled else "color: #999999;"
        self.title_label.setStyleSheet(f"font-weight: bold; {dim}")
        self.params_label.setStyleSheet(dim)

        glyph = _TOGGLE_ENABLED_GLYPH if enabled else _TOGGLE_DISABLED_GLYPH
        color = _TOGGLE_ENABLED_COLOR if enabled else _TOGGLE_DISABLED_COLOR
        self.toggle_button.setText(glyph)
        self.toggle_button.setStyleSheet(f"QToolButton {{ color: {color}; font-weight: bold; }}")
        tooltip = "Disable this filter (bypass)" if enabled else "Enable this filter"
        self.toggle_button.setToolTip(tooltip)
        self.toggle_button.setAccessibleName(f"{'Disable' if enabled else 'Enable'} FILT{self._position + 1}")

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self.clicked.emit(self.block_id)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not (event.buttons() & Qt.MouseButton.LeftButton) or self._drag_start is None:
            return
        moved = event.position().toPoint() - self._drag_start
        if moved.manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(MIME_BLOCK_ID, self.block_id.encode("utf-8"))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)
        self._drag_start = None
