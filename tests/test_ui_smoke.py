"""Qt smoke tests for the Phase 5 UI shell (CONTRACTS.md §14 phase-5 row).

CONTRACTS.md §14 says "manual/UI smoke only in this phase; no pytest
coverage expected for Qt widgets themselves" -- but the task explicitly
calls for a focused automated set, so these exercise `FilterCanvas` /
`MainWindow` directly (calling their public mutation methods, and
constructing drag/drop and key events programmatically) rather than
simulating real mouse-driven drag sequences, which need a native
windowing system that `QT_QPA_PLATFORM=offscreen` does not provide.

`QT_QPA_PLATFORM=offscreen` is set here (before any PyQt6 import) so the
whole suite runs headless; see `test_offscreen_platform_is_active` for an
explicit check that it took effect.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QMimeData, QPointF, Qt
from PyQt6.QtGui import QDropEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog, QLabel, QMenu, QMessageBox, QPushButton, QSplitter

from filters import FilterChain
from signals import SignalChain
from ui import app as app_module
from ui.app import DEFAULT_FS_HZ, MainWindow
from ui.canvas import FilterCanvas
from ui.signal_canvas import SignalCanvas
from ui.widgets.filter_block import MIME_BLOCK_ID


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    win = MainWindow()
    yield win
    # Explicit teardown: an undisposed QMainWindow (with its embedded
    # matplotlib FigureCanvases) left for Python's cyclic GC to collect at
    # an arbitrary later point -- e.g. mid-savefig() in an unrelated later
    # test -- has been observed to crash the interpreter (a PyQt6/sip +
    # matplotlib GC-timing interaction), once enough of these accumulate
    # across a test file. Closing and dropping the reference here keeps
    # each test's Qt object graph from piling up into the next one.
    # `mark_clean()` first so this teardown-time close() never raises a
    # real (unmocked, blocking) dirty-state confirmation dialog for a test
    # that intentionally left either chain dirty (closeEvent now checks
    # both, since the signal chain is also part of the project, §15).
    win.chain.mark_clean()
    win.time_domain_view.signal_chain.mark_clean()
    win.close()
    win.deleteLater()
    QApplication.processEvents()


@pytest.fixture
def canvas(qapp):
    chain = FilterChain(fs=13_333.0)
    widget = FilterCanvas(chain)
    yield widget
    widget.deleteLater()
    QApplication.processEvents()


@pytest.fixture
def signal_canvas(qapp):
    chain = SignalChain()
    widget = SignalCanvas(chain)
    yield widget
    widget.deleteLater()
    QApplication.processEvents()


# --- offscreen / startup -----------------------------------------------------


def test_offscreen_platform_is_active(qapp):
    assert QApplication.platformName() == "offscreen"


def test_application_startup(window):
    assert window.windowTitle().startswith("IIR Filter Design Suite")
    assert window.chain.blocks == []
    assert window.chain.fs == DEFAULT_FS_HZ
    assert window.export_action.isEnabled()


def test_no_validate_action_exists(window):
    """The Validate action was removed -- validation is automatic (CONTRACTS.md §13)."""
    assert not hasattr(window, "validate_action")
    assert not hasattr(window, "_on_validate")
    menu_texts = [action.text() for menu in window.menuBar().findChildren(QMenu) for action in menu.actions()]
    assert "Validate" not in menu_texts


# --- adding filter types ------------------------------------------------------


@pytest.mark.parametrize("kind", ["LP", "HP", "BP", "AP", "PK"])
def test_adding_each_filter_type(canvas, kind):
    block_id = canvas.add_block(kind)
    assert canvas.chain.get_block(block_id).kind == kind
    assert block_id in canvas._block_widgets
    assert canvas.chain.blocks[-1].id == block_id


# --- selection -----------------------------------------------------------------


def test_selecting_a_block(canvas):
    id1 = canvas.add_block("LP")
    id2 = canvas.add_block("HP")

    received = []
    canvas.selection_changed.connect(received.append)

    canvas.select_block(id2)

    assert canvas.chain.selected_id == id2
    assert canvas._block_widgets[id2].selected is True
    assert canvas._block_widgets[id1].selected is False
    assert received == [id2]


# --- reordering -----------------------------------------------------------------


def test_reordering(canvas):
    id1 = canvas.add_block("LP")
    id2 = canvas.add_block("HP")
    id3 = canvas.add_block("AP", Q=1.0)

    canvas.move_block(id3, 0)

    assert [b.id for b in canvas.chain.blocks] == [id3, id1, id2]


def test_reorder_via_drop_event(canvas):
    id1 = canvas.add_block("LP")
    id2 = canvas.add_block("HP")
    canvas.resize(300, 200)
    canvas.show()
    QApplication.processEvents()

    mime = QMimeData()
    mime.setData(MIME_BLOCK_ID, id2.encode("utf-8"))
    event = QDropEvent(
        QPointF(10, 0),
        Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.dropEvent(event)

    assert canvas.chain.blocks[0].id == id2


# --- delete-key removal -----------------------------------------------------------


def test_delete_key_removal(canvas):
    id1 = canvas.add_block("LP")
    id2 = canvas.add_block("HP")
    canvas.select_block(id1)

    QTest.keyClick(canvas, Qt.Key.Key_Delete)

    assert [b.id for b in canvas.chain.blocks] == [id2]
    assert canvas.chain.selected_id is None


def test_delete_key_noop_without_selection(canvas):
    canvas.add_block("LP")
    QTest.keyClick(canvas, Qt.Key.Key_Delete)
    assert len(canvas.chain.blocks) == 1


# --- clear confirmation -----------------------------------------------------------


def test_clear_confirmation_accept(canvas, monkeypatch):
    canvas.add_block("LP")
    canvas.add_block("HP")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    canvas.clear()

    assert canvas.chain.blocks == []


def test_clear_confirmation_decline(canvas, monkeypatch):
    canvas.add_block("LP")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)

    canvas.clear()

    assert len(canvas.chain.blocks) == 1


def test_clear_on_empty_chain_skips_confirmation(canvas, monkeypatch):
    calls = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: calls.append(1) or QMessageBox.StandardButton.Yes
    )

    canvas.clear()

    assert calls == []


# --- delete icon: confirms, unlike the Delete key -----------------------------


def test_delete_icon_requests_confirmation_and_removes_on_accept(canvas, monkeypatch):
    id1 = canvas.add_block("LP")
    id2 = canvas.add_block("HP")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    canvas._block_widgets[id1].delete_button.click()

    assert [b.id for b in canvas.chain.blocks] == [id2]


def test_delete_icon_declined_keeps_block(canvas, monkeypatch):
    id1 = canvas.add_block("LP")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)

    canvas._block_widgets[id1].delete_button.click()

    assert [b.id for b in canvas.chain.blocks] == [id1]


def test_delete_key_still_removes_without_confirmation(canvas, monkeypatch):
    """Keyboard Delete/Backspace stays unconfirmed -- only the icon path confirms."""
    id1 = canvas.add_block("LP")
    canvas.select_block(id1)
    calls = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: calls.append(1) or QMessageBox.StandardButton.Yes
    )

    QTest.keyClick(canvas, Qt.Key.Key_Delete)

    assert canvas.chain.blocks == []
    assert calls == []


def test_action_controls_are_inline_children_not_separate_windows(canvas):
    """Delete/toggle controls must live inside the block widget's own parent
    chain -- never surface as an independent top-level window (a stray
    QPushButton()/QToolButton() created before being embedded in a parented
    layout would otherwise flash as a "mini window")."""
    block_id = canvas.add_block("LP")
    widget = canvas._block_widgets[block_id]

    for button in (widget.delete_button, widget.toggle_button):
        assert button.parent() is widget
        assert button.isWindow() is False
        assert button.window() is widget.window()


def _flush_deferred_deletes():
    """`QApplication.processEvents()` alone does not reliably dispatch
    `QEvent.Type.DeferredDelete` (posted by `.deleteLater()`) in this Qt
    build -- confirmed by direct experiment. Qt's real event loop
    (`app.exec()`) flushes these naturally between user actions; tests need
    to force it explicitly instead."""
    from PyQt6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def test_replaced_block_widgets_do_not_linger_as_stray_windows(window):
    """`FilterCanvas.refresh()` rebuilds every block widget from `chain.blocks`
    on each model mutation, detaching the previous generation. A widget that
    is reparented to `None` while still visible becomes an independent
    top-level window (and on a real windowing system can be briefly mapped
    on screen by the window manager before Qt's own hidden-state suppresses
    it) -- `refresh()` must `hide()` widgets before detaching them, and
    `deleteLater()` them so they don't linger as zombie top-level QObjects
    (registered in `QApplication.topLevelWidgets()`) across further
    rebuilds. This is the exact bug reported: extra small windows, one per
    filter block, appearing as blocks are added to the workspace. Selecting
    a block between adds (as a user naturally would, clicking one block
    before dragging in the next) is what actually triggers the
    `Inspector`-driven double refresh that compounds this -- see
    `test_add_block_triggers_exactly_one_canvas_refresh`."""
    window.show()
    QApplication.processEvents()

    id1 = window.canvas.add_block("LP", fc=3000.0)
    window.canvas.select_block(id1)
    QApplication.processEvents()
    id2 = window.canvas.add_block("HP", fc=5000.0)
    window.canvas.select_block(id2)
    QApplication.processEvents()
    window.canvas.add_block("AP", fc=1000.0, Q=1.0)
    _flush_deferred_deletes()

    stray_blocks = [w for w in QApplication.topLevelWidgets() if w.objectName() == "filterBlock"]
    assert stray_blocks == []


def test_add_block_triggers_exactly_one_canvas_refresh(qapp, monkeypatch):
    """`Inspector.refresh()`'s tab rebuild used to unblock `QTabWidget`
    signals before its own `setCurrentIndex()` call. That's harmless while
    the current tab is "Combined" (index 0): `setCurrentIndex(0)` is a
    no-op since the index doesn't change. But once a real block tab is
    selected -- exactly what happens when a user clicks a block on the
    canvas -- the next structural refresh's `setCurrentIndex()` moves the
    index away from 0, which *does* fire a live `currentChanged` ->
    `block_selected` -> `canvas.refresh()` cascade: a second, redundant
    full rebuild of every `FilterBlockWidget` for a single model mutation,
    compounding how many stale widget generations `refresh()` has to tear
    down (see `test_replaced_block_widgets_do_not_linger_as_stray_windows`).
    The monkeypatch must be installed before `MainWindow` is constructed,
    since `chain_changed.connect(self.refresh)` binds to whatever `refresh`
    is at connection time."""
    from ui.canvas import FilterCanvas

    call_count = 0
    orig_refresh = FilterCanvas.refresh

    def counting_refresh(self):
        nonlocal call_count
        call_count += 1
        return orig_refresh(self)

    monkeypatch.setattr(FilterCanvas, "refresh", counting_refresh)

    win = MainWindow()
    try:
        block_id = win.canvas.add_block("LP")
        win.canvas.select_block(block_id)
        QApplication.processEvents()

        call_count = 0
        win.canvas.add_block("HP")
        assert call_count == 1
    finally:
        win.chain.mark_clean()
        win.close()
        win.deleteLater()
        QApplication.processEvents()


def test_action_controls_are_compact_and_fixed_size(canvas):
    from PyQt6.QtCore import QSize

    block_id = canvas.add_block("LP")
    widget = canvas._block_widgets[block_id]

    for button in (widget.delete_button, widget.toggle_button):
        assert button.minimumSize() == button.maximumSize() == QSize(20, 20)


def test_request_remove_block_unknown_id_is_a_noop(canvas, monkeypatch):
    calls = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: calls.append(1) or QMessageBox.StandardButton.Yes
    )

    canvas.request_remove_block("nope")  # must not raise

    assert calls == []


# --- enable/disable toggle icon ------------------------------------------------


def test_new_block_widget_starts_enabled(canvas):
    block_id = canvas.add_block("LP")
    widget = canvas._block_widgets[block_id]
    assert canvas.chain.get_block(block_id).enabled is True
    assert widget.enabled_state is True


def test_toggle_button_disables_and_reenables_block(canvas):
    block_id = canvas.add_block("LP")

    canvas._block_widgets[block_id].toggle_button.click()
    assert canvas.chain.get_block(block_id).enabled is False
    assert canvas._block_widgets[block_id].enabled_state is False

    canvas._block_widgets[block_id].toggle_button.click()
    assert canvas.chain.get_block(block_id).enabled is True
    assert canvas._block_widgets[block_id].enabled_state is True


def test_toggle_disable_does_not_remove_or_deselect_block(canvas):
    block_id = canvas.add_block("LP")
    canvas.select_block(block_id)

    canvas.toggle_block_enabled(block_id)

    assert block_id in [b.id for b in canvas.chain.blocks]
    assert canvas.chain.selected_id == block_id
    assert canvas._block_widgets[block_id].selected is True


def test_disabled_block_widget_shows_visual_indicator(canvas):
    block_id = canvas.add_block("LP")
    canvas.toggle_block_enabled(block_id)
    widget = canvas._block_widgets[block_id]

    assert "disabled" in widget.title_label.text().lower()
    assert widget.toggle_button.toolTip() == "Enable this filter"


def test_toggle_button_shows_green_symbol_when_enabled(canvas):
    block_id = canvas.add_block("LP")
    widget = canvas._block_widgets[block_id]

    assert "#27ae60" in widget.toggle_button.styleSheet()  # green
    assert widget.toggle_button.accessibleName() == "Disable FILT1"


def test_toggle_button_shows_grey_symbol_when_disabled(canvas):
    block_id = canvas.add_block("LP")
    canvas.toggle_block_enabled(block_id)
    widget = canvas._block_widgets[block_id]

    style = widget.toggle_button.styleSheet()
    assert "#27ae60" not in style  # no longer green
    assert widget.toggle_button.accessibleName() == "Enable FILT1"


def test_delete_button_reads_as_a_plain_x(canvas):
    block_id = canvas.add_block("LP")
    widget = canvas._block_widgets[block_id]

    assert widget.delete_button.text() == "✕"
    assert widget.delete_button.accessibleName() == "Delete FILT1"


def test_toggle_disable_does_not_trigger_confirmation(canvas, monkeypatch):
    block_id = canvas.add_block("LP")
    calls = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: calls.append(1) or QMessageBox.StandardButton.Yes
    )

    canvas._block_widgets[block_id].toggle_button.click()

    assert calls == []


def test_toggle_button_click_updates_model_view_and_export_status(window):
    """Clicking the inline toggle control (not calling the model directly)
    must update the `FilterChain` state, the block widget's own rendering,
    and the toolbar's Export enabled state in one round trip."""
    block_id = window.canvas.add_block("LP", fc=999_999.0)  # invalid while enabled
    assert not window.export_action.isEnabled()
    widget = window.canvas._block_widgets[block_id]

    widget.toggle_button.click()

    assert window.chain.get_block(block_id).enabled is False  # model
    refreshed_widget = window.canvas._block_widgets[block_id]
    assert refreshed_widget.enabled_state is False  # view (rebuilt by refresh())
    assert window.export_action.isEnabled()  # export status: invalid-but-disabled no longer blocks


def test_disabled_invalid_block_reenables_export(window):
    block_id = window.canvas.add_block("LP", fc=999_999.0)
    assert not window.export_action.isEnabled()

    window.canvas.set_block_enabled(block_id, False)

    assert window.export_action.isEnabled()


def test_disabling_and_deleting_are_independent(canvas):
    id1 = canvas.add_block("LP")
    id2 = canvas.add_block("HP")

    canvas.toggle_block_enabled(id1)
    canvas.remove_block(id2)

    assert [b.id for b in canvas.chain.blocks] == [id1]
    assert canvas.chain.get_block(id1).enabled is False


def test_clear_preserves_fs(window, monkeypatch):
    window.fs_edit.setText("20000")
    window.fs_edit.editingFinished.emit()
    window.canvas.add_block("LP")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    window.canvas.clear()

    assert window.chain.blocks == []
    assert window.chain.fs == 20_000.0


# --- invalid block display -----------------------------------------------------


def test_invalid_block_display(canvas):
    block_id = canvas.add_block("LP", fc=999_999.0)
    widget = canvas._block_widgets[block_id]

    assert not canvas.chain.get_block(block_id).is_valid
    assert not widget.error_label.isHidden()
    assert widget.error_label.text() == canvas.chain.get_block(block_id).error


def test_validate_export_disabled_with_invalid_block(window):
    window.canvas.add_block("LP", fc=999_999.0)

    assert not window.export_action.isEnabled()


def test_validate_export_reenabled_after_fix(window):
    block_id = window.canvas.add_block("LP", fc=999_999.0)
    assert not window.export_action.isEnabled()

    window.chain.update_params(block_id, fc=1000.0)
    window.canvas.refresh()
    window._on_chain_changed()

    assert window.export_action.isEnabled()


# --- fs validation -----------------------------------------------------------------


def test_fs_validation_rejects_out_of_range(window):
    window.fs_edit.setText("999999")
    window.fs_edit.editingFinished.emit()

    assert not window.fs_error_label.isHidden()
    assert window.chain.fs == DEFAULT_FS_HZ


def test_fs_validation_accepts_valid_value(window):
    window.canvas.add_block("LP", fc=1000.0)
    window.fs_edit.setText("20000")
    window.fs_edit.editingFinished.emit()

    assert window.fs_error_label.isHidden()
    assert window.chain.fs == 20_000.0


def test_fs_change_invalidates_out_of_range_block(window):
    block_id = window.canvas.add_block("LP", fc=15_000.0)  # valid at default fs=13333
    window.fs_edit.setText("5000")
    window.fs_edit.editingFinished.emit()

    assert not window.chain.get_block(block_id).is_valid
    assert not window.export_action.isEnabled()


# --- reset ------------------------------------------------------------------------


def test_reset_restores_defaults(window, monkeypatch):
    window.fs_edit.setText("20000")
    window.fs_edit.editingFinished.emit()
    window.canvas.add_block("LP")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    window._on_reset()

    assert window.chain.blocks == []
    assert window.chain.fs == DEFAULT_FS_HZ
    assert window.chain.dirty is False


def test_reset_declined_keeps_state(window, monkeypatch):
    window.canvas.add_block("LP")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)

    window._on_reset()

    assert len(window.chain.blocks) == 1


# --- dirty-state quit warning -------------------------------------------------------


def test_dirty_state_quit_warning_cancelled_keeps_window_open(window, monkeypatch):
    # Automatic validation (CONTRACTS.md §13) re-runs `refresh_validation()`
    # after every mutation and marks the chain clean again immediately, so
    # `add_block()` alone no longer leaves `chain.dirty` True. Set it
    # directly to exercise the quit-warning code path itself.
    window.canvas.add_block("LP")
    window.chain.dirty = True
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)

    window.show()
    closed = window.close()

    assert closed is False
    assert window.isVisible()
    window.chain.mark_clean()  # avoid a real modal dialog when the fixture drops the window


def test_dirty_state_quit_warning_discarded_closes_window(window, monkeypatch):
    window.canvas.add_block("LP")
    window.chain.dirty = True
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard)

    window.show()
    closed = window.close()

    assert closed is True


def test_dirty_state_quit_warning_save_closes_window(window, monkeypatch, tmp_path):
    window.canvas.add_block("LP")
    window.chain.dirty = True
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save)
    window._project_path = tmp_path / "quit_save.iirfilt"

    window.show()
    closed = window.close()

    assert closed is True
    assert window.chain.dirty is False
    assert window._project_path.exists()


def test_dirty_state_quit_warning_save_as_cancelled_keeps_window_open(window, monkeypatch):
    # Save chosen but no project path yet -- `_on_save` falls through to
    # `_on_save_as`, whose file dialog the user then cancels. The chain
    # stays dirty, so the close must be aborted rather than discarding.
    window.canvas.add_block("LP")
    window.chain.dirty = True
    window._project_path = None
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))

    window.show()
    closed = window.close()

    assert closed is False
    assert window.isVisible()
    window.chain.mark_clean()  # avoid a real modal dialog when the fixture drops the window


def test_clean_state_quits_without_warning(window, monkeypatch):
    calls = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: calls.append(1) or QMessageBox.StandardButton.Save
    )

    window.show()
    closed = window.close()

    assert closed is True
    assert calls == []


# --- compiler-failure Retry/Quit dialog (CONTRACTS.md §9, §13) -------------------------


def test_ensure_native_backend_retry_then_success(qapp, monkeypatch):
    calls = {"n": 0}

    def fake_get(force_retry: bool = False):
        calls["n"] += 1
        if calls["n"] == 1:
            assert force_retry is False
            return None, "boom: fake compiler stderr (first ~20 lines)"
        assert force_retry is True
        return "FAKE_BACKEND", None

    monkeypatch.setattr(app_module, "_get_shared_native_backend", fake_get)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: self.buttons()[0])  # Retry is added first

    result = app_module.ensure_native_backend()

    assert result == "FAKE_BACKEND"
    assert calls["n"] == 2  # one failed attempt, one retry that succeeded


def test_ensure_native_backend_quit_returns_none_without_raising(qapp, monkeypatch):
    monkeypatch.setattr(app_module, "_get_shared_native_backend", lambda force_retry=False: (None, "boom"))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: self.buttons()[1])  # Quit is added second

    result = app_module.ensure_native_backend()

    assert result is None


def test_main_returns_nonzero_and_shows_no_window_when_compiler_unavailable(qapp, monkeypatch):
    # QApplication is a process-wide singleton; the session already has one
    # (`qapp`), so main()'s own `QApplication(argv)` construction is
    # monkeypatched to hand back that same instance rather than constructing
    # a second one (undefined behavior in Qt -- observed to segfault).
    monkeypatch.setattr(app_module, "QApplication", lambda argv: qapp)
    monkeypatch.setattr(app_module, "ensure_native_backend", lambda: None)
    created = []
    monkeypatch.setattr(app_module, "MainWindow", lambda *a, **k: created.append(1))

    exit_code = app_module.main([])

    assert exit_code != 0
    assert created == []  # Quit exits before any MainWindow is constructed


# --- Export wiring (CONTRACTS.md §10, §13) --------------------------------------------


def test_export_action_writes_full_file_set(window, monkeypatch, tmp_path):
    window.canvas.add_block("LP", fc=3000.0)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    info_calls = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info_calls.append(a[2] if len(a) > 2 else ""))

    window._on_export()

    export_dirs = list(tmp_path.iterdir())
    assert len(export_dirs) == 1
    assert (export_dirs[0] / "design.iirfilt").is_file()
    assert (export_dirs[0] / "source" / "biquad_q14" / "gen" / "filter_design.h").is_file()
    assert (export_dirs[0] / "reports" / "biquad_q14_report.pdf").is_file()

    # Success is reported via a confirmation dialog naming the export dir.
    assert len(info_calls) == 1
    assert str(export_dirs[0]) in info_calls[0]


def test_export_cancelled_dialog_writes_nothing(window, monkeypatch, tmp_path):
    window.canvas.add_block("LP", fc=3000.0)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")

    window._on_export()

    assert list(tmp_path.iterdir()) == []


def test_export_empty_chain_reports_error_without_raising(window, monkeypatch, tmp_path):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    critical_calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: critical_calls.append(a[2] if len(a) > 2 else ""))

    window._on_export()  # must not raise -- empty-chain ValueError is caught and reported

    assert len(critical_calls) == 1
    assert "empty" in critical_calls[0]
    # No export directory was ever created for this failure, so the error
    # dialog falls back to reporting the destination folder the user picked.
    assert str(tmp_path) in critical_calls[0]
    assert list(tmp_path.iterdir()) == []


def test_export_pdf_failure_reports_error_with_partial_export_dir(window, monkeypatch, tmp_path):
    """A failure raised after the export directory is already created (e.g.
    PDF generation) must point the error dialog at that specific directory,
    not just the destination folder the user picked."""
    window.canvas.add_block("LP", fc=3000.0)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    critical_calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: critical_calls.append(a[2] if len(a) > 2 else ""))

    import export as export_module

    monkeypatch.setattr(
        export_module,
        "render_pdf",
        lambda *a, **k: (_ for _ in ()).throw(export_module.ExportError("boom")),
    )

    window._on_export()

    assert len(critical_calls) == 1
    assert "boom" in critical_calls[0]
    export_dirs = list(tmp_path.iterdir())
    assert len(export_dirs) == 1  # the export dir was created before the injected failure
    assert str(export_dirs[0]) in critical_calls[0]


def test_export_always_forces_fresh_validation_not_a_stale_cache(window, monkeypatch, tmp_path):
    block_id = window.canvas.add_block("LP", fc=1000.0)
    window.inspector.refresh_validation()  # caches Inspector-side metrics for fc=1000
    # Mutates the model directly, bypassing the Inspector's own edit path,
    # so its cached metrics above go stale -- export must never read them.
    window.chain.update_params(block_id, fc=5000.0)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    window._on_export()

    export_dir = next(tmp_path.iterdir())
    header_text = (export_dir / "source" / "biquad_q14" / "gen" / "filter_design.h").read_text()
    assert "fc = 5000 Hz" in header_text
    assert "fc = 1000 Hz" not in header_text


def test_export_includes_time_domain_artifacts_when_signal_blocks_present(window, monkeypatch, tmp_path):
    """`_on_export()` wires the Time-Domain view's own SignalChain plus its
    Inspector's current duration/full-scale fields into export_design()
    (CONTRACTS.md §10's time-domain export addendum) -- exercised here via
    the real UI-facing defaults, not by calling export_design() directly."""
    window.canvas.add_block("LP", fc=3000.0)
    window.time_domain_view.canvas.add_block("SIN")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    window._on_export()

    export_dir = next(tmp_path.iterdir())
    assert (export_dir / "reports" / "figures" / "time_domain_combined.png").is_file()
    assert (export_dir / "reports" / "figures" / "time_domain_lp_1.png").is_file()
    assert (export_dir / "reports" / "data" / "time_domain_combined.csv").is_file()

    from project_file import load_project

    _, _, signal_blocks = load_project(export_dir / "design.iirfilt")
    assert [b["kind"] for b in signal_blocks] == ["SIN"]


def test_export_omits_time_domain_artifacts_when_no_signal_blocks(window, monkeypatch, tmp_path):
    window.canvas.add_block("LP", fc=3000.0)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    window._on_export()

    export_dir = next(tmp_path.iterdir())
    figure_names = {p.name for p in (export_dir / "reports" / "figures").iterdir()}
    assert not any(name.startswith("time_domain_") for name in figure_names)
    assert not (export_dir / "reports" / "data").exists()


# --- Project file (save/open, CONTRACTS.md §15) -----------------------------------------


def test_save_as_writes_project_file(window, monkeypatch, tmp_path):
    window.canvas.add_block("LP", fc=3000.0)
    target = tmp_path / "myproject.iirfilt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))

    window._on_save_as()

    assert target.is_file()
    assert window._project_path == target
    assert window.chain.dirty is False


def test_save_as_cancelled_dialog_writes_nothing(window, monkeypatch, tmp_path):
    window.canvas.add_block("LP")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))

    window._on_save_as()

    assert list(tmp_path.iterdir()) == []
    assert window._project_path is None


def test_save_reuses_existing_path_without_dialog(window, monkeypatch, tmp_path):
    window.canvas.add_block("LP")
    target = tmp_path / "myproject.iirfilt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window._on_save_as()

    calls = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: calls.append(1) or (str(target), ""))
    window.canvas.add_block("HP")

    window._on_save()

    assert calls == []  # no dialog on a plain Save once a path is known
    assert window.chain.dirty is False


def test_open_replaces_chain_in_place(window, monkeypatch, tmp_path):
    """Open must mutate the existing FilterChain, never swap in a new one --
    canvas/inspector hold a reference to the original instance."""
    from project_file import save_project

    original_chain = window.chain
    source = FilterChain(fs=22_050.0)
    source.add_block("HP", fc=500.0)
    path = tmp_path / "other.iirfilt"
    save_project(source, path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))

    window._on_open()

    assert window.chain is original_chain
    assert window.canvas.chain is original_chain
    assert window.chain.fs == 22_050.0
    assert [b.kind for b in window.chain.blocks] == ["HP"]
    assert window.chain.dirty is False


def test_open_confirms_when_dirty_and_declines_keeps_state(window, monkeypatch, tmp_path):
    window.canvas.add_block("LP")
    window.chain.dirty = True
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    dialog_calls = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: dialog_calls.append(1) or ("", ""))

    window._on_open()

    assert dialog_calls == []  # never even opened the file dialog
    assert len(window.chain.blocks) == 1


def test_open_malformed_file_shows_error_without_crashing(window, monkeypatch, tmp_path):
    bad = tmp_path / "bad.iirfilt"
    bad.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(bad), ""))
    critical_calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: critical_calls.append(a[2] if len(a) > 2 else ""))

    window._on_open()  # must not raise

    assert len(critical_calls) == 1


def test_open_cancelled_dialog_is_a_noop(window, monkeypatch):
    window.canvas.add_block("LP")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))

    window._on_open()

    assert len(window.chain.blocks) == 1


# --- signal-chain persistence (schema v2, §11/§15) --------------------------------------


def test_save_as_then_open_round_trips_signal_chain(window, monkeypatch, tmp_path):
    window.time_domain_view.canvas.add_block("SIN")
    window.time_domain_view.signal_chain.update_params(
        window.time_domain_view.signal_chain.blocks[0].id, frequency=2_000.0
    )
    target = tmp_path / "with_signals.iirfilt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window._on_save_as()
    assert window.time_domain_view.signal_chain.dirty is False

    # Mutate further, then re-open the just-saved file to prove Open rebuilds
    # the *same* SignalChain instance (canvas/inspector hold a reference to
    # it) from the saved signal blocks, mirroring the filter-chain contract.
    window.time_domain_view.canvas.add_block("DC")  # leaves signal_chain dirty again
    original_signal_chain = window.time_domain_view.signal_chain
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(target), ""))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    window._on_open()

    assert window.time_domain_view.signal_chain is original_signal_chain
    assert [b.kind for b in window.time_domain_view.signal_chain.blocks] == ["SIN"]
    assert window.time_domain_view.signal_chain.blocks[0].params["frequency"] == 2_000.0
    assert window.time_domain_view.signal_chain.dirty is False


def test_signal_chain_edit_marks_title_dirty(window):
    assert "*" not in window.windowTitle()

    window.time_domain_view.canvas.add_block("DC")

    assert "*" in window.windowTitle()
    window.time_domain_view.signal_chain.mark_clean()  # avoid a dialog when the fixture closes the window


def test_open_confirms_when_only_signal_chain_is_dirty(window, monkeypatch):
    window.time_domain_view.canvas.add_block("DC")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    dialog_calls = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: dialog_calls.append(1) or ("", ""))

    window._on_open()

    assert dialog_calls == []  # never even opened the file dialog
    assert len(window.time_domain_view.signal_chain.blocks) == 1


def test_close_warns_when_only_signal_chain_is_dirty(window, monkeypatch):
    window.time_domain_view.canvas.add_block("DC")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard)

    window.show()
    closed = window.close()

    assert closed is True


# --- Noise reseed control (docs/CONCEPT.md §11) -----------------------------------------


def test_reseed_button_present_only_for_noise_blocks(signal_canvas):
    signal_canvas.add_block("DC")
    assert signal_canvas.findChild(QPushButton, "reseedButton") is None

    signal_canvas.add_block("NOISE")
    assert signal_canvas.findChild(QPushButton, "reseedButton") is not None


def test_reseed_button_click_draws_a_new_seed_and_stays_valid(signal_canvas):
    block_id = signal_canvas.add_block("NOISE")
    seed_before = signal_canvas.signal_chain.get_block(block_id).design.seed

    signal_canvas.findChild(QPushButton, "reseedButton").click()

    block = signal_canvas.signal_chain.get_block(block_id)
    assert block.is_valid, block.error
    assert block.design.seed != seed_before


def test_reseed_button_click_rerenders_seed_label(signal_canvas):
    signal_canvas.add_block("NOISE")

    signal_canvas.findChild(QPushButton, "reseedButton").click()

    block = signal_canvas.signal_chain.blocks[0]
    seed_label = signal_canvas.findChild(QLabel, "signalBlockSeed")
    assert seed_label.text() == str(block.params["seed"])


def test_reseed_button_click_does_not_disturb_amplitude_field(signal_canvas):
    block_id = signal_canvas.add_block("NOISE")
    signal_canvas.update_params(block_id, {"amplitude": 0.3})

    signal_canvas.findChild(QPushButton, "reseedButton").click()

    assert signal_canvas.signal_chain.blocks[0].params["amplitude"] == 0.3


def test_reseed_request_for_unknown_block_is_a_noop(signal_canvas):
    block_id = signal_canvas.add_block("NOISE")
    seed_before = signal_canvas.signal_chain.get_block(block_id).design.seed

    signal_canvas.reseed_block("no-such-id")  # must not raise

    assert signal_canvas.signal_chain.get_block(block_id).design.seed == seed_before


# --- adjustable workspace layout (QSplitter) --------------------------------------------


def test_main_window_uses_a_splitter_for_the_central_layout(window):
    assert isinstance(window.splitter, QSplitter)
    assert window.splitter.orientation() == Qt.Orientation.Horizontal
    assert window.splitter.count() == 3


def test_workspace_pane_opens_narrower_than_the_inspector_pane(window):
    sizes = window.splitter.sizes()
    assert sizes[1] < sizes[2]  # workspace (index 1) < inspector (index 2)


def test_splitter_panes_are_user_resizable(window):
    original = window.splitter.sizes()
    total = sum(original)

    window.splitter.setSizes([total // 3, total // 3, total - 2 * (total // 3)])

    assert window.splitter.sizes() != list(original)


def test_palette_and_canvas_still_reachable_through_the_splitter(window):
    assert window.splitter.widget(0) is window.palette
    block_id = window.canvas.add_block("LP")
    assert block_id in window.canvas._block_widgets


def test_filter_cards_remain_selectable_in_splitter_layout(window):
    block_id = window.canvas.add_block("LP")

    window.canvas.select_block(block_id)

    assert window.canvas._block_widgets[block_id].selected is True


def test_inspector_tabs_remain_present_in_splitter_layout(window):
    window.canvas.add_block("LP")
    assert window.inspector.tabs.count() == 2
