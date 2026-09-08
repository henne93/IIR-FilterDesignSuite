"""PyQt6 application shell (Phase 5, CONTRACTS.md §13).

Owns the chain-wide `fs` input, the File/Edit menu bar, and dirty-state
tracking. `FilterChain` (src/python/filters/chain.py) remains
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
from PyQt6.QtGui import QAction, QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from c_codegen import NativeBackend as _CNativeBackend
from export import ExportError, export_design
from filters import FilterChain
from project_file import PROJECT_FILE_EXTENSION, ProjectFileError, load_project, save_project
from ui.canvas import FilterCanvas
from ui.inspector import Inspector
from ui.palette import FilterPalette
from ui.time_domain_view import TimeDomainView

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
        self._project_path: Path | None = None
        self.canvas = FilterCanvas(self.chain)
        self.palette = FilterPalette()
        # `main()` compiles the backend via `ensure_native_backend()` before
        # any MainWindow is constructed, so this is normally an instant
        # cache hit. The fallback lookup only matters for tests/tools that
        # construct MainWindow() directly without going through main().
        self.backend = backend if backend is not None else _get_shared_native_backend()[0]
        self.inspector = Inspector(self.chain, self.backend)

        self._build_actions()
        self._build_menu_bar()
        self._build_central_widget()
        self.statusBar()

        self.canvas.chain_changed.connect(self._on_chain_changed)
        self.canvas.chain_changed.connect(self.inspector.refresh)
        self.canvas.chain_changed.connect(self.time_domain_view.refresh_fs)
        self.canvas.selection_changed.connect(self.inspector.select_block)
        self.inspector.block_selected.connect(lambda _block_id: self.canvas.refresh())
        self.inspector.params_changed.connect(self._on_inspector_params_changed)
        self.inspector.params_changed.connect(self.time_domain_view.refresh_fs)
        self._on_chain_changed()

    # -- layout ---------------------------------------------------------

    def _build_central_widget(self) -> None:
        design_view = QWidget()
        outer = QVBoxLayout(design_view)

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

        # Second top-level view (docs/CONCEPT.md §11.7): swaps the whole
        # central widget rather than adding a 4th panel to the Design view.
        self.time_domain_view = TimeDomainView(self.chain, self.backend)

        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(design_view)
        self.view_stack.addWidget(self.time_domain_view)
        self.setCentralWidget(self.view_stack)
        self._build_view_switch()

    def _build_view_switch(self) -> None:
        """Design/Time Domain toggle, docked in the menu bar's corner (CONCEPT.md §11.7 mockup)."""
        switch = QWidget()
        switch_layout = QHBoxLayout(switch)
        switch_layout.setContentsMargins(0, 0, 8, 0)

        self.design_view_button = QPushButton("Design")
        self.time_domain_view_button = QPushButton("Time Domain")
        for button in (self.design_view_button, self.time_domain_view_button):
            button.setCheckable(True)
            switch_layout.addWidget(button)

        self.view_switch_group = QButtonGroup(self)
        self.view_switch_group.setExclusive(True)
        self.view_switch_group.addButton(self.design_view_button, 0)
        self.view_switch_group.addButton(self.time_domain_view_button, 1)
        self.design_view_button.setChecked(True)
        self.view_switch_group.idClicked.connect(self.view_stack.setCurrentIndex)

        self.menuBar().setCornerWidget(switch)

    def _build_actions(self) -> None:
        self.open_action = QAction("Open", self)
        self.open_action.triggered.connect(self._on_open)

        self.save_action = QAction("Save", self)
        self.save_action.triggered.connect(self._on_save)

        self.save_as_action = QAction("Save As", self)
        self.save_as_action.triggered.connect(self._on_save_as)

        self.export_action = QAction("Export", self)
        self.export_action.triggered.connect(self._on_export)

        self.clear_action = QAction("Clear", self)
        self.clear_action.triggered.connect(self.canvas.clear)

        self.reset_action = QAction("Reset", self)
        self.reset_action.triggered.connect(self._on_reset)

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)

        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self.clear_action)
        edit_menu.addAction(self.reset_action)

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
        self.time_domain_view.refresh_fs()
        self._on_chain_changed()

    # -- Project file (save/open, CONTRACTS.md §15) --------------------------

    def _on_save(self) -> None:
        if self._project_path is None:
            self._on_save_as()
            return
        self._save_to(self._project_path)

    def _on_save_as(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save project as", str(Path.cwd()), f"IIR Filter Project (*{PROJECT_FILE_EXTENSION})"
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix != PROJECT_FILE_EXTENSION:
            path = path.with_name(path.name + PROJECT_FILE_EXTENSION)
        self._save_to(path)

    def _save_to(self, path: Path) -> None:
        try:
            save_project(self.chain, path)
        except ProjectFileError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._project_path = path
        self.chain.mark_clean()
        self._update_title()
        self.statusBar().showMessage(f"Saved to {path}", 5000)

    def _on_open(self) -> None:
        if self.chain.dirty:
            reply = QMessageBox.question(
                self,
                "Open project?",
                "The filter chain has unsaved changes. Open a different project anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open project", str(Path.cwd()), f"IIR Filter Project (*{PROJECT_FILE_EXTENSION})"
        )
        if not path_str:
            return

        try:
            fs, blocks = load_project(path_str)
        except ProjectFileError as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return  # load fully before touching the chain -- a bad file never leaves it half-mutated

        try:
            self.chain.fs = fs
        except ValueError as exc:
            QMessageBox.critical(self, "Open failed", f"invalid project file: {exc}")
            return

        self.chain.clear()
        for block in blocks:
            block_id = self.chain.add_block(block["kind"], **block["params"])
            if not block["enabled"]:
                self.chain.set_enabled(block_id, False)
        self.chain.mark_clean()
        self._project_path = Path(path_str)

        self.fs_edit.setText(f"{self.chain.fs:g}")
        self.fs_error_label.setVisible(False)
        self.canvas.refresh()
        self.inspector.refresh()
        self.time_domain_view.refresh_fs()
        self._on_chain_changed()
        self.statusBar().showMessage(f"Opened {path_str}", 5000)

    # -- Export -----------------------------------------------------------------

    def _on_export(self) -> None:
        """Wired to `export.export_design()` (CONTRACTS.md §10, §13).

        `export_design()` always performs its own independent fresh
        validation pass internally (never reads the Inspector's own
        metrics), and raises `ValueError` for the two conditions export
        refuses to proceed past -- an empty chain or any invalid block --
        both reported here rather than left as a traceback. Filesystem/
        plotting/PDF failures surface as `ExportError`, also reported here.
        Both error dialogs include the relevant path (the export directory
        if one is known, otherwise the destination folder the user picked).
        On success, a confirmation dialog reports the export directory too.
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
            # ExportError may know the specific export directory it failed
            # inside (see export.ExportError/export_design); a plain
            # ValueError (invalid chain) never got that far, so falls back
            # to the destination folder the user picked.
            location = getattr(exc, "output_dir", None) or Path(output_root)
            QMessageBox.critical(self, "Export failed", f"{exc}\n\nLocation: {location}")
            return

        QMessageBox.information(
            self, "Export successful", f"Export completed successfully.\n\nLocation: {result.output_dir}"
        )
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
        self.time_domain_view.refresh_fs()
        self._on_chain_changed()

    # -- close / dirty warning -----------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not self.chain.dirty:
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            "The filter chain has unsaved changes. Save before closing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if reply == QMessageBox.StandardButton.Save:
            self._on_save()
            if self.chain.dirty:
                # Save-as was cancelled or save_project() failed -- stay open
                # rather than discarding changes the user asked to keep.
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
