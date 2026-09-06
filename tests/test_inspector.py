"""Qt smoke tests for the Phase 5 Inspector (CONTRACTS.md §12, §13).

Mirrors `test_ui_smoke.py`'s approach: exercises `Inspector`/`MainWindow`
through their public methods and by driving widgets programmatically
(setting QLineEdit text + emitting `editingFinished`) rather than simulating
real mouse-driven interaction, which needs a native windowing system that
`QT_QPA_PLATFORM=offscreen` does not provide.

Uses the session-scoped `native_backend` fixture from `conftest.py` (a real
ctypes-backed `NativeBackend`, compiled once for the whole test session) so
Q14 plotting/coefficient-table/metrics assertions exercise the real
quantization path, not a stand-in.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from error_analysis import linear_response_grid
from filters import FilterChain
from filters.base import fc_max
from ui.app import MainWindow
from ui.inspector import BlockInspectorPanel, CombinedInspectorPanel, Inspector
from ui.widgets.gain_widget import GainWidget


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _motion_event(ax, xdata):
    """Fabricates a matplotlib-motion-event-shaped object for direct handler calls."""
    return SimpleNamespace(inaxes=ax, xdata=xdata)


@pytest.fixture
def chain(qapp) -> FilterChain:
    return FilterChain(fs=13_333.0)


@pytest.fixture
def inspector(qapp, chain, native_backend) -> Inspector:
    return Inspector(chain, native_backend)


def _edit(field, text: str) -> None:
    field.setText(text)
    field.editingFinished.emit()


# --- Combined + per-block tabs -----------------------------------------------------


def test_combined_tab_always_present_and_first(inspector):
    assert inspector.tabs.count() == 1
    assert inspector.tabs.tabText(0) == "Combined"
    assert isinstance(inspector._panels[None], CombinedInspectorPanel)


@pytest.mark.parametrize("kind", ["LP", "HP", "BP", "AP"])
def test_per_block_tab_added_for_each_kind(inspector, chain, kind):
    block_id = chain.add_block(kind)
    inspector.refresh()

    assert inspector.tabs.count() == 2
    assert inspector.tabs.tabText(1) == f"FILT1: {kind}"
    assert isinstance(inspector._panels[block_id], BlockInspectorPanel)


def test_tab_titles_renumber_on_reorder(inspector, chain):
    id1 = chain.add_block("LP")
    id2 = chain.add_block("HP")
    inspector.refresh()
    chain.move_block(id2, 0)
    inspector.refresh()

    assert inspector.tabs.tabText(1) == "FILT1: HP"
    assert inspector.tabs.tabText(2) == "FILT2: LP"


def test_removing_selected_block_falls_back_to_combined_tab(inspector, chain):
    block_id = chain.add_block("LP")
    inspector.refresh()
    inspector.select_block(block_id)
    assert inspector.tabs.currentIndex() == 1

    chain.remove_block(block_id)
    inspector.refresh()

    assert inspector.tabs.count() == 1
    assert inspector.tabs.currentIndex() == 0
    assert chain.selected_id is None


# --- LP/HP/BP/AP forms -----------------------------------------------------------------


def test_lp_form_has_fc_field(inspector, chain):
    block_id = chain.add_block("LP", fc=2_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    assert set(panel._fields.keys()) == {"fc"}
    assert panel._fields["fc"].text() == "2000"


def test_hp_form_has_fc_field(inspector, chain):
    block_id = chain.add_block("HP", fc=3_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    assert set(panel._fields.keys()) == {"fc"}
    assert panel._fields["fc"].text() == "3000"


def test_bp_form_has_f_low_f_high_fields(inspector, chain):
    block_id = chain.add_block("BP", f_low=1_000.0, f_high=2_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    assert set(panel._fields.keys()) == {"f_low", "f_high"}
    assert panel._fields["f_low"].text() == "1000"
    assert panel._fields["f_high"].text() == "2000"


def test_ap_form_has_fc_and_q_fields(inspector, chain):
    block_id = chain.add_block("AP", fc=1_000.0, Q=1.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    assert set(panel._fields.keys()) == {"fc", "Q"}


# --- BP derived values -----------------------------------------------------------------


def test_bp_derived_fc_and_q_readonly_display(inspector, chain):
    block_id = chain.add_block("BP", f_low=1_000.0, f_high=4_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    derived = chain.get_block(block_id).filter.derived_params()
    assert f"{derived['fc']:.4g}" in panel._derived_labels["fc"].text()
    assert f"{derived['Q']:.4g}" in panel._derived_labels["Q"].text()
    # read-only: no QLineEdit exists for fc/Q on a BP panel
    assert "fc" not in panel._fields
    assert "Q" not in panel._fields


def test_bp_derived_values_update_after_edit(inspector, chain):
    block_id = chain.add_block("BP", f_low=1_000.0, f_high=2_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]
    before = panel._derived_labels["fc"].text()

    _edit(panel._fields["f_high"], "4000")

    after = panel._derived_labels["fc"].text()
    assert after != before
    assert chain.get_block(block_id).filter.derived_params()["fc"] > 0


def test_bp_derived_values_blank_when_invalid(inspector, chain):
    block_id = chain.add_block("BP", f_low=1_000.0, f_high=2_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    _edit(panel._fields["f_high"], "500")  # f_high < f_low -> invalid

    assert not chain.get_block(block_id).is_valid
    assert panel._derived_labels["fc"].text() == "—"
    assert panel._derived_labels["Q"].text() == "—"


# --- AP Q validation -----------------------------------------------------------------


def test_ap_invalid_q_shows_inline_error_without_crashing(inspector, chain):
    block_id = chain.add_block("AP", fc=1_000.0, Q=1.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    _edit(panel._fields["Q"], "10.0")  # Q_MAX is 4.0

    assert not chain.get_block(block_id).is_valid
    assert not panel.error_label.isHidden()
    assert "Q" in panel.error_label.text()


def test_ap_q_fixed_clears_error(inspector, chain):
    block_id = chain.add_block("AP", fc=1_000.0, Q=1.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    _edit(panel._fields["Q"], "10.0")
    assert panel.error_label.text() != ""

    _edit(panel._fields["Q"], "2.0")

    assert chain.get_block(block_id).is_valid
    assert panel.error_label.isHidden()


def test_ap_non_numeric_q_rejected_without_touching_model(inspector, chain):
    block_id = chain.add_block("AP", fc=1_000.0, Q=1.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    _edit(panel._fields["Q"], "not-a-number")

    assert chain.get_block(block_id).params["Q"] == 1.0  # model untouched
    assert not panel.error_label.isHidden()


# --- parameter updates through the model -----------------------------------------------


def test_editing_fc_field_routes_through_chain_update_params(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    _edit(panel._fields["fc"], "4000")

    assert chain.get_block(block_id).params["fc"] == 4000.0
    assert chain.get_block(block_id).filter.fc == 4000.0


def test_editing_field_emits_params_changed(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]
    received = []
    inspector.params_changed.connect(lambda: received.append(1))

    _edit(panel._fields["fc"], "4000")

    assert received == [1]


def test_editing_field_immediately_revalidates_and_stays_clean(inspector, chain):
    """Automatic validation (CONTRACTS.md §13): no manual Validate step, so an
    edit's own `refresh_validation()` call leaves the chain clean again
    immediately -- it never observably goes "dirty until Validate."
    """
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    assert chain.dirty is False
    panel = inspector._panels[block_id]

    _edit(panel._fields["fc"], "4000")

    assert chain.dirty is False


# --- ideal/Q14 plot rendering -----------------------------------------------------------


def test_bode_plot_renders_ideal_and_q14_curves(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    assert len(panel.bode.ax_mag.lines) == 2  # ideal + Q14
    assert len(panel.bode.ax_phase.lines) == 2


def test_bode_plot_ideal_only_when_backend_none(chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector = Inspector(chain, backend=None)
    panel = inspector._panels[block_id]

    assert len(panel.bode.ax_mag.lines) == 1  # ideal only


def test_combined_bode_plot_renders_after_adding_blocks(inspector, chain):
    chain.add_block("LP", fc=2_000.0)
    chain.add_block("HP", fc=500.0)
    inspector.refresh()
    combined = inspector._panels[None]

    assert len(combined.bode.ax_mag.lines) == 2


# --- coefficient table values -----------------------------------------------------------


def test_coefficient_table_shows_ideal_and_q14_values(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]
    filt = chain.get_block(block_id).filter
    ideal = filt.ideal_coefficients()
    q14 = filt.q14_coefficients(inspector.backend)

    assert float(panel.table.item(0, 0).text()) == pytest.approx(ideal.b0, abs=1e-6)
    assert int(panel.table.item(0, 1).text()) == q14.b0
    assert float(panel.table.item(0, 2).text()) == pytest.approx(q14.b0 / 16384, abs=1e-6)


def test_coefficient_table_cleared_when_block_invalid(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    _edit(panel._fields["fc"], "999999")  # out of range

    assert panel.table.item(0, 0).text() == ""


# --- response metrics -----------------------------------------------------------------


def test_refresh_validation_computes_per_block_response_error(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()

    inspector.refresh_validation()

    panel = inspector._panels[block_id]
    assert "Response error vs Q14" in panel.metrics.response_label.text()
    assert "Coefficient sweep" in panel.metrics.sweep_label.text()


def test_refresh_validation_computes_combined_response_error(inspector, chain):
    chain.add_block("LP", fc=2_000.0)
    chain.add_block("HP", fc=500.0)
    inspector.refresh()

    inspector.refresh_validation()

    combined = inspector._panels[None]
    assert "Response error vs Q14" in combined.metrics.response_label.text()
    assert combined.metrics.sweep_label.text() == ""  # no per-coefficient sweep for Combined


def test_bp_block_shows_coefficient_sweep_and_response_error(inspector, chain):
    block_id = chain.add_block("BP", f_low=1_000.0, f_high=2_000.0)
    inspector.refresh()

    inspector.refresh_validation()

    panel = inspector._panels[block_id]
    assert "Response error vs Q14" in panel.metrics.response_label.text()
    assert "Coefficient sweep" in panel.metrics.sweep_label.text()
    sweep = inspector._sweep_cache[block_id]
    assert 100.0 <= sweep.worst_frequency_hz <= fc_max(chain.fs)


def test_refresh_validation_marks_chain_clean(inspector, chain):
    chain.add_block("LP", fc=1_000.0)
    assert chain.dirty is True

    inspector.refresh_validation()

    assert chain.dirty is False


# --- automatic validation: metrics are always fresh, never "stale" ---------------------


def test_metrics_visible_immediately_after_refresh(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()

    panel = inspector._panels[block_id]
    assert panel.metrics.status_label.isHidden() is True
    assert panel.metrics.response_label.styleSheet() == ""


def test_edit_immediately_produces_fresh_metrics_no_staleness(inspector, chain):
    """There is no manual Validate step, so an edit's automatic
    `refresh_validation()` pass replaces the metrics synchronously -- they
    are never shown greyed-out/"stale" (CONTRACTS.md §13, revised)."""
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]
    result_text_before = panel.metrics.response_label.text()

    _edit(panel._fields["fc"], "1500")

    assert panel.metrics.status_label.isHidden() is True
    assert panel.metrics.response_label.styleSheet() == ""
    assert panel.metrics.response_label.text() != result_text_before
    assert "Response error vs Q14" in panel.metrics.response_label.text()


def test_metrics_recomputed_fresh_after_structural_refresh(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    result_text = inspector._panels[block_id].metrics.response_label.text()

    chain.add_block("HP", fc=500.0)  # unrelated structural change
    inspector.refresh()

    panel = inspector._panels[block_id]
    assert panel.metrics.response_label.text() == result_text  # unaffected, still fresh
    assert panel.metrics.status_label.isHidden() is True  # never shown stale


# --- invalid and empty-chain states -----------------------------------------------------


def test_empty_chain_shows_explicit_message_no_crash(inspector):
    combined = inspector._panels[None]
    assert "No filter blocks" in combined.status_label.text()
    assert len(combined.bode.ax_mag.lines) == 0


def test_empty_chain_refresh_validation_does_not_crash(inspector):
    inspector.refresh_validation()  # must not raise
    combined = inspector._panels[None]
    assert "add at least one filter block" in combined.metrics.status_label.text().lower()


def test_all_invalid_chain_combined_shows_message(inspector, chain):
    chain.add_block("LP", fc=999_999.0)
    chain.add_block("HP", fc=-5.0)
    inspector.refresh()

    combined = inspector._panels[None]
    assert "all" in combined.status_label.text().lower()
    assert "invalid" in combined.status_label.text().lower()
    assert len(combined.bode.ax_mag.lines) == 0


def test_all_invalid_chain_refresh_validation_does_not_crash(inspector, chain):
    chain.add_block("LP", fc=999_999.0)
    inspector.refresh()

    inspector.refresh_validation()  # must not raise

    combined = inspector._panels[None]
    assert "invalid" in combined.metrics.status_label.text().lower()


def test_partially_invalid_chain_combined_excludes_invalid_blocks(inspector, chain):
    chain.add_block("LP", fc=1_000.0)
    chain.add_block("HP", fc=999_999.0)
    inspector.refresh()

    combined = inspector._panels[None]
    assert "1 of 2" in combined.status_label.text()
    assert len(combined.bode.ax_mag.lines) == 2  # still plots using the valid block


def test_invalid_block_panel_shows_message_not_plot(inspector, chain):
    block_id = chain.add_block("LP", fc=999_999.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    assert not panel.error_label.isHidden()
    assert len(panel.bode.ax_mag.lines) == 0


# --- enabled/disabled (bypass) blocks -----------------------------------------------


def test_disabled_block_excluded_from_combined_tab(inspector, chain):
    chain.add_block("LP", fc=1_000.0)
    hp_id = chain.add_block("HP", fc=1_000.0)
    chain.set_enabled(hp_id, False)
    inspector.refresh()

    combined = inspector._panels[None]
    assert "1 of 2" in combined.status_label.text()
    assert "disabled" in combined.status_label.text().lower()
    assert len(combined.bode.ax_mag.lines) == 2  # still plots using the enabled block


def test_all_disabled_chain_combined_shows_message(inspector, chain):
    bid = chain.add_block("LP", fc=1_000.0)
    chain.set_enabled(bid, False)
    inspector.refresh()

    combined = inspector._panels[None]
    assert "disabled" in combined.status_label.text().lower()
    assert len(combined.bode.ax_mag.lines) == 0


def test_disabled_block_still_shows_its_own_tab_and_plot(inspector, chain):
    """Disabled only bypasses the *combined* analysis -- the block's own tab stays live."""
    block_id = chain.add_block("LP", fc=1_000.0)
    chain.set_enabled(block_id, False)
    inspector.refresh()

    panel = inspector._panels[block_id]
    assert panel.error_label.isHidden()
    assert len(panel.bode.ax_mag.lines) > 0


def test_refresh_validation_marks_chain_clean_with_disabled_block(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    chain.set_enabled(block_id, False)
    inspector.refresh()

    inspector.refresh_validation()  # must not raise

    assert chain.dirty is False


# --- MainWindow integration ---------------------------------------------------------


@pytest.fixture
def window(qapp):
    return MainWindow()


def test_inspector_present_in_mainwindow(window):
    assert window.inspector.tabs.count() == 1  # Combined only, empty chain


def test_canvas_add_block_rebuilds_inspector_tabs(window):
    window.canvas.add_block("LP")
    assert window.inspector.tabs.count() == 2


def test_canvas_selection_syncs_to_inspector_tab(window):
    block_id = window.canvas.add_block("LP")
    window.canvas.select_block(block_id)

    assert window.inspector.tabs.currentIndex() == 1


def test_inspector_tab_selection_syncs_to_canvas_highlight(window):
    block_id = window.canvas.add_block("LP")
    window.inspector.select_block(block_id)

    assert window.canvas._block_widgets[block_id].selected is True


def test_inspector_field_edit_updates_canvas_summary(window):
    block_id = window.canvas.add_block("LP", fc=1_000.0)
    panel = window.inspector._panels[block_id]

    _edit(panel._fields["fc"], "3000")

    assert "3000" in window.canvas._block_widgets[block_id].params_label.text()


def test_inspector_field_edit_updates_toolbar_enabled_state(window):
    block_id = window.canvas.add_block("LP", fc=1_000.0)
    panel = window.inspector._panels[block_id]

    _edit(panel._fields["fc"], "999999")

    assert not window.export_action.isEnabled()


def test_toggle_button_click_syncs_canvas_inspector_and_export_status(window):
    """Clicking the inline enable/disable control on the canvas block widget
    must reach the model, the canvas's own re-render, the Inspector's
    Combined tab, and the toolbar's Export enabled state -- all four in
    sync, not just the model."""
    block_id = window.canvas.add_block("LP", fc=1_000.0)
    second_id = window.canvas.add_block("HP", fc=999_999.0)  # invalid while enabled
    assert not window.export_action.isEnabled()
    widget = window.canvas._block_widgets[second_id]

    widget.toggle_button.click()

    assert window.chain.get_block(second_id).enabled is False
    assert window.export_action.isEnabled()
    combined = window.inspector._panels[None]
    assert "1 of 2" in combined.status_label.text()
    assert "disabled" in combined.status_label.text().lower()


def test_adding_block_automatically_computes_metrics_without_a_validate_step(window):
    """No manual Validate action exists -- adding a block alone must produce
    fresh metrics via `Inspector.refresh_validation()` (CONTRACTS.md §13)."""
    window.canvas.add_block("LP", fc=1_000.0)

    assert window.chain.dirty is False
    block_id = window.chain.blocks[0].id
    panel = window.inspector._panels[block_id]
    assert "Response error vs Q14" in panel.metrics.response_label.text()


def test_fs_change_refreshes_inspector_bode_grid(window):
    block_id = window.canvas.add_block("LP", fc=6_000.0)  # valid at fs=13333
    window.fs_edit.setText("20000")
    window.fs_edit.editingFinished.emit()

    panel = window.inspector._panels[block_id]
    assert panel.error_label.isHidden() is True
    assert len(panel.bode.ax_mag.lines) == 2


def test_fs_change_invalidating_block_reflects_in_inspector(window):
    block_id = window.canvas.add_block("LP", fc=15_000.0)  # valid at default fs
    window.fs_edit.setText("5000")
    window.fs_edit.editingFinished.emit()

    panel = window.inspector._panels[block_id]
    assert panel.error_label.isHidden() is False


# --- automatic validation trigger points (CONTRACTS.md §13) ----------------------------


def test_param_edit_automatically_updates_metrics(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]
    before = panel.metrics.response_label.text()

    _edit(panel._fields["fc"], "3000")

    after = panel.metrics.response_label.text()
    assert after != before
    assert "Response error vs Q14" in after


def test_adding_filter_triggers_validation(window):
    block_id = window.canvas.add_block("LP", fc=1_000.0)

    panel = window.inspector._panels[block_id]
    assert "Response error vs Q14" in panel.metrics.response_label.text()


def test_removing_filter_triggers_validation(window):
    window.canvas.add_block("LP", fc=1_000.0)
    id2 = window.canvas.add_block("HP", fc=500.0)

    window.canvas.remove_block(id2)

    combined = window.inspector._panels[None]
    assert "Response error vs Q14" in combined.metrics.response_label.text()


def test_reordering_filters_triggers_validation(window):
    id1 = window.canvas.add_block("LP", fc=2_000.0)
    id2 = window.canvas.add_block("HP", fc=500.0)

    window.canvas.move_block(id2, 0)

    combined = window.inspector._panels[None]
    assert "Response error vs Q14" in combined.metrics.response_label.text()


def test_clearing_filters_triggers_validation(window, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    window.canvas.add_block("LP", fc=1_000.0)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    window.canvas.clear()

    combined = window.inspector._panels[None]
    assert "add at least one filter block" in combined.metrics.status_label.text().lower()


def test_sample_rate_change_triggers_validation(window):
    block_id = window.canvas.add_block("LP", fc=6_000.0)

    window.fs_edit.setText("20000")
    window.fs_edit.editingFinished.emit()

    panel = window.inspector._panels[block_id]
    assert "Response error vs Q14" in panel.metrics.response_label.text()


# --- linear gain plot (added below the Bode plot in every tab) -------------------------


def test_block_panel_has_gain_widget(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()

    assert isinstance(inspector._panels[block_id].gain, GainWidget)


def test_combined_panel_has_gain_widget(inspector):
    assert isinstance(inspector._panels[None].gain, GainWidget)


def test_gain_plot_renders_ideal_and_q14_curves(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    assert len(panel.gain.ax_gain.lines) == 2  # ideal + Q14


def test_gain_plot_ideal_only_when_backend_none(chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector = Inspector(chain, backend=None)
    panel = inspector._panels[block_id]

    assert len(panel.gain.ax_gain.lines) == 1


def test_gain_plot_axes_are_linear(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    assert panel.gain.ax_gain.get_xscale() == "linear"
    assert panel.gain.ax_gain.get_yscale() == "linear"


def test_gain_values_equal_db_to_gain_conversion(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    filt = chain.get_block(block_id).filter
    lin_freq = linear_response_grid(chain.fs)
    ideal_resp = filt.ideal_response(lin_freq)
    expected_gain = 10.0 ** (ideal_resp.magnitude_db / 20.0)

    ideal_line = panel.gain.ax_gain.lines[0]
    np.testing.assert_allclose(ideal_line.get_ydata(), expected_gain)


def test_combined_tab_gain_plot_renders_both_curves(inspector, chain):
    chain.add_block("LP", fc=2_000.0)
    chain.add_block("HP", fc=500.0)
    inspector.refresh()

    combined = inspector._panels[None]
    assert len(combined.gain.ax_gain.lines) == 2


def test_gain_plot_shows_message_for_invalid_block(inspector, chain):
    block_id = chain.add_block("LP", fc=999_999.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    assert len(panel.gain.ax_gain.lines) == 0


def test_gain_plot_shows_message_for_empty_chain(inspector):
    combined = inspector._panels[None]
    assert len(combined.gain.ax_gain.lines) == 0


# --- shared hover measurement cursor (ui/widgets/measurement_cursor.py) ----------------


def test_bode_motion_updates_both_widgets_cursor(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    panel.cursor._on_bode_motion(_motion_event(panel.bode.ax_mag, 1000.0))

    assert len(panel.bode._cursor_artists) > 0
    assert len(panel.gain._cursor_artists) > 0


def test_gain_motion_updates_bode_widget_cursor(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    panel.cursor._on_gain_motion(_motion_event(panel.gain.ax_gain, 1000.0))

    assert len(panel.bode._cursor_artists) > 0
    assert len(panel.gain._cursor_artists) > 0


def test_cursor_uses_the_same_frequency_in_both_widgets(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    panel.cursor._on_bode_motion(_motion_event(panel.bode.ax_mag, 1234.0))

    bode_x = panel.bode._cursor_artists[0].get_segments()[0][0][0]
    gain_x = panel.gain._cursor_artists[0].get_segments()[0][0][0]
    assert bode_x == pytest.approx(1234.0)
    assert gain_x == pytest.approx(1234.0)


def test_cursor_annotation_shows_ideal_and_q14_values(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    panel.cursor._on_bode_motion(_motion_event(panel.bode.ax_mag, 1000.0))

    bode_annotation = panel.bode._cursor_artists[-1].get_text()
    assert "Ideal" in bode_annotation
    assert "Q14" in bode_annotation

    gain_annotation = panel.gain._cursor_artists[-1].get_text()
    assert "Ideal" in gain_annotation
    assert "Q14" in gain_annotation


def test_cursor_ignores_events_outside_the_axes(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    panel.cursor._on_bode_motion(_motion_event(None, 1000.0))

    assert panel.bode._cursor_artists == []
    assert panel.gain._cursor_artists == []


def test_cursor_ignores_events_with_no_xdata(inspector, chain):
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    panel.cursor._on_bode_motion(_motion_event(panel.bode.ax_mag, None))

    assert panel.bode._cursor_artists == []
    assert panel.gain._cursor_artists == []


def test_cursor_does_not_activate_for_an_invalid_block(inspector, chain):
    block_id = chain.add_block("LP", fc=999_999.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    panel.cursor._on_bode_motion(_motion_event(panel.bode.ax_mag, 1000.0))  # must not raise

    assert panel.bode._cursor_artists == []


def test_cursor_state_independent_between_tabs(inspector, chain):
    id1 = chain.add_block("LP", fc=1_000.0)
    id2 = chain.add_block("HP", fc=500.0)
    inspector.refresh()
    panel1 = inspector._panels[id1]
    panel2 = inspector._panels[id2]

    panel1.cursor._on_bode_motion(_motion_event(panel1.bode.ax_mag, 1000.0))

    assert panel1.bode._cursor_artists != []
    assert panel2.bode._cursor_artists == []


def test_cursor_curve_count_unaffected_by_active_cursor(inspector, chain):
    """Cursor artists use `Axes.vlines()` (-> `ax.collections`), not
    `ax.lines`, so an active cursor must not change response-curve counts."""
    block_id = chain.add_block("LP", fc=1_000.0)
    inspector.refresh()
    panel = inspector._panels[block_id]

    panel.cursor._on_bode_motion(_motion_event(panel.bode.ax_mag, 1000.0))

    assert len(panel.bode.ax_mag.lines) == 2
    assert len(panel.gain.ax_gain.lines) == 2
