"""Single signal-block widget rendered on the Time-Domain canvas (docs/CONCEPT.md §11.7).

Unlike `FilterBlockWidget` (whose params are edited via the Inspector's
per-block tabs), a signal block has no dedicated per-block Inspector tab --
§11.6's tab strip is keyed to *filter* chain blocks, not signal blocks -- so
its parameters are edited directly on this canvas tile via inline fields.

Ownership rule (mirrors filter_block.py): this widget never mutates
`SignalChain` directly. It only emits `params_edited`/`delete_requested`;
`SignalCanvas` is the one that calls into the model and re-renders.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from signals import SignalChainBlock

_KIND_NAMES = {"SIN": "Sine", "DC": "DC", "NOISE": "Noise", "CSV": "CSV Import"}
_ACTION_BUTTON_SIZE = QSize(20, 20)


class SignalBlockWidget(QFrame):
    params_edited = pyqtSignal(str, dict)  # block_id, {param: value} (value: float | str)
    delete_requested = pyqtSignal(str)  # block_id

    def __init__(self, block: SignalChainBlock, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.block_id = block.id
        self.setObjectName("signalBlock")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel(_KIND_NAMES.get(block.kind, block.kind))
        title.setStyleSheet("font-weight: bold;")
        header.addWidget(title, 1)

        self.delete_button = QToolButton(self)
        self.delete_button.setObjectName("deleteSignalBlockButton")
        self.delete_button.setAutoRaise(True)
        self.delete_button.setFixedSize(_ACTION_BUTTON_SIZE)
        self.delete_button.setText("✕")
        self.delete_button.setToolTip("Delete this signal block")
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self.block_id))
        header.addWidget(self.delete_button)
        layout.addLayout(header)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self._fields: dict[str, QLineEdit] = {}
        self._build_kind_fields(form, block)
        self._fields["factor"] = self._add_field(form, "factor", "factor", block.factor)
        layout.addLayout(form)

        self.error_label = QLabel(block.error or "")
        self.error_label.setObjectName("signalBlockError")
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(not block.is_valid)
        layout.addWidget(self.error_label)

        self._set_border(block.is_valid)

    # -- form construction ---------------------------------------------------

    def _build_kind_fields(self, form: QFormLayout, block: SignalChainBlock) -> None:
        kind = block.kind
        if kind == "SIN":
            self._fields["frequency"] = self._add_field(form, "freq (Hz)", "frequency", block.params["frequency"])
            self._fields["amplitude"] = self._add_field(form, "amplitude", "amplitude", block.params["amplitude"])
            self._fields["phase_deg"] = self._add_field(form, "phase (°)", "phase_deg", block.params["phase_deg"])
        elif kind == "DC":
            self._fields["value"] = self._add_field(form, "value", "value", block.params["value"])
        elif kind == "NOISE":
            self._fields["amplitude"] = self._add_field(form, "amplitude", "amplitude", block.params["amplitude"])
        elif kind == "CSV":
            file_path = str(block.params.get("file_path", ""))
            self.file_label = QLabel(Path(file_path).name if file_path else "(no file)")
            self.file_label.setWordWrap(True)
            form.addRow("file", self.file_label)
            browse = QPushButton("Browse…")
            browse.clicked.connect(self._on_browse_csv)
            form.addRow("", browse)

    def _add_field(self, form: QFormLayout, label: str, key: str, value: float) -> QLineEdit:
        edit = QLineEdit(f"{value:g}")
        edit.setObjectName(f"signalField_{key}")
        edit.editingFinished.connect(lambda k=key, e=edit: self._on_field_edited(k, e))
        form.addRow(label, edit)
        return edit

    def _on_field_edited(self, key: str, edit: QLineEdit) -> None:
        text = edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            self.error_label.setText(f"{key} must be a number, got {text!r}.")
            self.error_label.setVisible(True)
            return
        self.params_edited.emit(self.block_id, {key: value})

    def _on_browse_csv(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV files (*.csv);;All files (*)")
        if not path_str:
            return
        self.params_edited.emit(self.block_id, {"file_path": path_str})

    def _set_border(self, valid: bool) -> None:
        border = "1px solid #999999" if valid else "1px solid #c0392b"
        self.setStyleSheet(f"#signalBlock {{ border: {border}; border-radius: 4px; }}")
