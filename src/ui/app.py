"""PyQt6 application shell (Phase 5, CONTRACTS.md §13).

Owns the chain-wide `fs` input, the Export/Clear/Reset toolbar, and
dirty-state tracking. `FilterChain` (src/python/filters/chain.py) remains
the single source of truth -- this module never duplicates filter equations
or keeps independent widget-side parameters (CONTRACTS.md §12).

There is no Validate action: the `Inspector` (`ui/inspector.py`) revalidates
automatically after every model mutation (param edits, add/remove/reorder,
clear/reset, `fs` changes) via `Inspector.refresh_validation()`, so its
response-error / coefficient-sweep metrics are always current. Export
remains disabled while any *enabled* block in the chain is invalid
(CONTRACTS.md §5) -- a disabled block is a bypass and never gates Export,
even when invalid (`FilterChain.has_invalid_blocks`). Export is wired to
`export.export_design()`, which always performs its own independent fresh
validation pass regardless of the Inspector's state.

The central widget is a `QSplitter` (palette / filter-chain workspace /
inspector) so the user can resize the workspace pane; it opens narrower
than the inspector by default since the inspector now hosts both the Bode
and linear-gain plots.

Startup native-library compilation (CONTRACTS.md §9, §13): there is no
degraded/mock mode -- Q14 is load-bearing everywhere (Inspector, Export).
`main()` compiles the shared library via `ensure_native_backend()` *before*
any `MainWindow` is constructed or shown; on failure this blocks with a
modal Retry/Quit dialog showing the compiler's stderr excerpt. Retry
re-attempts compilation; Quit exits the process cleanly (no traceback,
before any Qt window exists).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from c_codegen import NativeBackend as _CNativeBackend
from export import ExportError, export_design
from filters import FilterChain
from ui.canvas import FilterCanvas
from ui.inspector import Inspector
from ui.palette import FilterPalette

DEFAULT_FS_HZ = 13_333.0

# Process-local NativeBackend singleton (CONTRACTS.md §9: "compiled once at
# app start, cached until the next launch" -- the process is that boundary,
# not any one MainWindow instance, so multiple windows in the same process
# share a single compile). Compilation is attempted lazily, once, on first
# use; a failure is cached too so repeated construction doesn't retry a
# doomed compile without an explicit `force_retry` (the dialog's Retry
# button in `ensure_native_backend()`).
_native_backend: _CNativeBackend | None = None
_native_backend_error: str | None = None


def _get_shared_native_backend(force_retry: bool = False) -> tuple[_CNativeBackend | None, str | None]:
    global _native_backend, _native_backend_error
    if force_retry:
        _native_backend_error = None
    if _native_backend is None and _native_backend_error is None:
        try:
            _native_backend = _CNativeBackend()
        except Exception as exc:  # CompilerNotFoundError / NativeCompileError, etc.
            _native_backend_error = str(exc)
    return _native_backend, _native_backend_error


def ensure_native_backend(parent: QWidget | None = None) -> _CNativeBackend | None:
    """Compiles the native Q14 backend before any window exists (CONTRACTS.md §9/§13).

    On failure, blocks with a modal dialog showing the compiler's stderr
    excerpt (already truncated to ~20 lines by `compile_shared_library`) and
    **Retry**/**Quit** buttons. Retry re-attempts compilation; Quit returns
    `None` so the caller can exit the process without ever showing a
    `MainWindow` and without raising -- no traceback on the Quit path.
    """
    backend, error = _get_shared_native_backend()
    while error is not None:
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Native library compilation failed")
        box.setText(
            "The Q14 native library failed to compile. The application cannot "
            "run without it (Q14 coefficients are required by the Inspector "
            "and Export)."
        )
        box.setInformativeText(error)
        retry_button = box.addButton("Retry", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Quit", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(retry_button)
        box.exec()
        if box.clickedButton() is not retry_button:
            return None
        backend, error = _get_shared_native_backend(force_retry=True)
    return backend


class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None, backend: _CNativeBackend | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("IIR Filter Design Suite")
        self.resize(900, 600)

        self.chain = FilterChain(fs=DEFAULT_FS_HZ)
        self.canvas = FilterCanvas(self.chain)
        self.palette = FilterPalette()
        # `main()` compiles the backend via `ensure_native_backend()` before
        # any MainWindow is constructed, so this is normally an instant
        # cache hit. The fallback lookup only matters for tests/tools that
        # construct MainWindow() directly without going through main().
        self.backend = backend if backend is not None else _get_shared_native_backend()[0]
        self.inspector = Inspector(self.chain, self.backend)

        self._build_toolbar()
        self._build_central_widget()
        self.statusBar()

        self.canvas.chain_changed.connect(self._on_chain_changed)
        self.canvas.chain_changed.connect(self.inspector.refresh)
        self.canvas.selection_changed.connect(self.inspector.select_block)
        self.inspector.block_selected.connect(lambda _block_id: self.canvas.refresh())
        self.inspector.params_changed.connect(self._on_inspector_params_changed)
        self._on_chain_changed()

    # -- layout ---------------------------------------------------------

    def _build_central_widget(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)

        fs_row = QHBoxLayout()
        fs_row.addWidget(QLabel("Sample rate fs (Hz):"))
        self.fs_edit = QLineEdit(f"{self.chain.fs:g}")
        self.fs_edit.setValidator(QDoubleValidator(0.0, 1.0e9, 6))
        self.fs_edit.setMaximumWidth(120)
        self.fs_edit.editingFinished.connect(self._on_fs_submitted)
        fs_row.addWidget(self.fs_edit)
        self.fs_error_label = QLabel("")
        self.fs_error_label.setStyleSheet("color: #c0392b;")
        self.fs_error_label.setVisible(False)
        fs_row.addWidget(self.fs_error_label)
        fs_row.addStretch(1)
        outer.addLayout(fs_row)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.palette)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        self.splitter.addWidget(scroll)
        self.splitter.addWidget(self.inspector)
        # Workspace (index 1) opens narrower than the inspector (index 2),
        # which now hosts both the Bode and linear-gain plots -- all three
        # panes stay user-resizable via the splitter handles.
        self.splitter.setSizes([160, 220, 520])
        outer.addWidget(self.splitter, 1)

        self.setCentralWidget(central)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.export_action = toolbar.addAction("Export")
        self.export_action.triggered.connect(self._on_export)

        toolbar.addSeparator()

        self.clear_action = toolbar.addAction("Clear")
        self.clear_action.triggered.connect(self.canvas.clear)

        self.reset_action = toolbar.addAction("Reset")
        self.reset_action.triggered.connect(self._on_reset)

    # -- reactions to model changes -----------------------------------------

    def _on_chain_changed(self) -> None:
        invalid = self.chain.has_invalid_blocks
        self.export_action.setEnabled(not invalid)
        self._update_title()

    def _update_title(self) -> None:
        star = "*" if self.chain.dirty else ""
        self.setWindowTitle(f"IIR Filter Design Suite{star}")

    def _on_inspector_params_changed(self) -> None:
        self.canvas.refresh()
        self._on_chain_changed()

    # -- fs -------------------------------------------------------------

    def _on_fs_submitted(self) -> None:
        text = self.fs_edit.text().strip()
        try:
            value = float(text)
            self.chain.fs = value
        except ValueError as exc:
            self.fs_error_label.setText(str(exc))
            self.fs_error_label.setVisible(True)
            return
        self.fs_error_label.setVisible(False)
        self.canvas.refresh()
        self.inspector.refresh()
        self._on_chain_changed()

    # -- Export -----------------------------------------------------------------

    def _on_export(self) -> None:
        """Wired to `export.export_design()` (CONTRACTS.md §10, §13).

        `export_design()` always performs its own independent fresh
        validation pass internally (never reads the Inspector's own
        metrics), and raises `ValueError` for the two conditions export
        refuses to proceed past -- an empty chain or any invalid block --
        both reported here rather than left as a traceback. Filesystem/
        plotting/PDF failures surface as `ExportError`, also reported here.
        """
        if self.backend is None:
            QMessageBox.critical(
                self,
                "Export failed",
                "The native Q14 backend is unavailable, so Q14 coefficients and "
                "metrics cannot be computed. Export requires it.",
            )
            return

        output_root = QFileDialog.getExistingDirectory(self, "Select export destination folder", str(Path.cwd()))
        if not output_root:
            return

        try:
            result = export_design(self.chain, self.backend, output_root)
        except (ValueError, ExportError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        self.statusBar().showMessage(f"Exported to {result.output_dir}", 5000)

    # -- reset ------------------------------------------------------------

    def _on_reset(self) -> None:
        if self.chain.blocks or self.chain.fs != DEFAULT_FS_HZ:
            reply = QMessageBox.question(
                self,
                "Reset filter chain?",
                "Reset clears every filter block and restores the default sample rate. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.chain.clear()
        self.chain.fs = DEFAULT_FS_HZ
        self.chain.mark_clean()
        self.fs_edit.setText(f"{DEFAULT_FS_HZ:g}")
        self.fs_error_label.setVisible(False)
        self.canvas.refresh()
        self.inspector.refresh()
        self._on_chain_changed()

    # -- close / dirty warning -----------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.chain.dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved changes",
                "The filter chain has unsaved changes. Quit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)

    # CONTRACTS.md §9/§13: compile before the main window is shown; no
    # degraded mode -- Quit exits here, before any MainWindow exists.
    backend = ensure_native_backend()
    if backend is None:
        return 1

    window = MainWindow(backend=backend)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
