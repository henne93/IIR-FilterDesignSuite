"""Inspector: Combined + per-block tabs with forms, Bode/gain plots, coefficient
table, and response/coefficient-sweep metrics (Phase 5, CONTRACTS.md §12/§13).

`FilterChain` remains the single source of truth (CONTRACTS.md §12): every
panel here reads `chain.blocks`/`chain.get_block(...)` fresh on each render
and routes edits through `chain.update_params()` -- nothing in this module
keeps a private copy of filter parameters. All response/coefficient-error
numbers come from `error_analysis.py` (`response_error`,
`combined_response_error`, `coefficient_sweep`, `bode_grid`,
`linear_response_grid`); no analysis math is duplicated here.

Automatic validation (CONTRACTS.md §13): there is no manual Validate action
anywhere in this UI. The Bode plot, the linear gain plot, the coefficient
table, and BP's derived fc/Q are cheap and always live -- recomputed on
every edit via `refresh_live()`. The response-error and coefficient-sweep
numbers are the more expensive artifacts; they are recomputed automatically
by `Inspector.refresh_validation()` after every model mutation (a field
edit's `editingFinished`, add/remove/reorder, clear/reset, or an `fs`
change) -- never cached across an edit, so metrics are never shown "stale."
`export.py`'s independent, always-fresh validation pass on top of this is
unaffected and still exists as a defensive boundary.

Each tab (Combined + one per block) owns one `MeasurementCursor`
(`ui/widgets/measurement_cursor.py`) synchronizing its Bode and gain plots'
hover measurement line. Cursor state is scoped to that tab's panel instance
and is never shared across tabs -- switching tabs never moves another tab's
cursor.

Band-Pass has a coefficient-sweep metric like every other kind (CONTRACTS.md
§6.3): it sweeps its center frequency while holding its configured
bandwidth (`f_high - f_low`) fixed, clamping `f_low`/`f_high` at the
`[100, fc_max(fs)]` sweep domain's edges whenever a symmetric window around
the center would otherwise leave that domain -- the same clamp strategy
`export.py::_sweep_design_at` uses and
tests/test_error_analysis.py::test_coefficient_sweep_bp_holds_bandwidth_fixed
validates.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from error_analysis import (
    COEFFICIENT_SWEEP_FLOOR_HZ,
    CoefficientSweepError,
    ResponseError,
    bode_grid,
    coefficient_sweep,
    combined_response_error,
    linear_response_grid,
    response_error,
)
from filters import ChainBlock, FilterChain
from filters.base import NativeBackend, fc_max
from ui.widgets.bode_widget import BodeWidget
from ui.widgets.gain_widget import GainWidget
from ui.widgets.measurement_cursor import MeasurementCursor

_COEFFICIENT_NAMES = ("b0", "b1", "b2", "a1", "a2")


class _MetricsPanel(QWidget):
    """Live max/RMS response error (+ optional coefficient-sweep stats).

    `set_result`/`set_unavailable` are pushed in by `Inspector` after every
    automatic `refresh_validation()` pass -- there is no manual Validate
    step and therefore no stale/greyed-out state to track.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.status_label = QLabel("")
        self.status_label.setObjectName("metricsStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        self.response_label = QLabel("")
        self.response_label.setObjectName("metricsResponse")
        layout.addWidget(self.response_label)

        self.sweep_label = QLabel("")
        self.sweep_label.setObjectName("metricsSweep")
        self.sweep_label.setWordWrap(True)
        layout.addWidget(self.sweep_label)

    def set_result(self, response: ResponseError, sweep: CoefficientSweepError | None) -> None:
        self.response_label.setText(
            f"Response error vs Q14 — max: {response.max_db:.4f} dB, RMS: {response.rms_db:.4f} dB"
        )
        if sweep is not None:
            per_coeff = ", ".join(f"{name}={sweep.per_coefficient_max[name]:.3e}" for name in _COEFFICIENT_NAMES)
            self.sweep_label.setText(
                f"Coefficient sweep — max abs: {sweep.max_abs:.3e}, RMS abs: {sweep.rms_abs:.3e}, "
                f"worst at {sweep.worst_frequency_hz:.1f} Hz ({per_coeff})"
            )
        else:
            self.sweep_label.setText("")
        self.status_label.setVisible(False)

    def set_unavailable(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_label.setVisible(True)
        self.response_label.setText("")
        self.sweep_label.setText("")


class BlockInspectorPanel(QWidget):
    """One chain block's tab: validated form, live Bode/gain plots, coefficient table, metrics."""

    params_changed = pyqtSignal()

    def __init__(
        self, chain: FilterChain, block_id: str, backend: NativeBackend | None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.chain = chain
        self.block_id = block_id
        self.backend = backend
        self._current_responses = None  # (ideal, q14) FrequencyResponse pair for the cursor, or None

        outer = QVBoxLayout(self)

        self.error_label = QLabel("")
        self.error_label.setObjectName("inspectorBlockError")
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        outer.addWidget(self.error_label)

        form = QFormLayout()
        self._fields: dict[str, QLineEdit] = {}
        self._derived_labels: dict[str, QLabel] = {}
        self._build_form(form, chain.get_block(block_id))
        outer.addLayout(form)

        self.bode = BodeWidget()
        outer.addWidget(self.bode, 1)

        self.gain = GainWidget()
        outer.addWidget(self.gain, 1)

        self.cursor = MeasurementCursor(self.bode, self.gain, lambda: self._current_responses)

        self.table = QTableWidget(len(_COEFFICIENT_NAMES), 3)
        self.table.setObjectName("coefficientTable")
        self.table.setHorizontalHeaderLabels(["Ideal (float64)", "Q14 (int16)", "Q14 (float)"])
        self.table.setVerticalHeaderLabels(list(_COEFFICIENT_NAMES))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        outer.addWidget(self.table)

        self.metrics = _MetricsPanel()
        outer.addWidget(self.metrics)

        self.refresh_live()

    # -- form construction ---------------------------------------------------

    def _build_form(self, form: QFormLayout, block: ChainBlock) -> None:
        kind = block.kind
        if kind in ("LP", "HP"):
            self._fields["fc"] = self._add_field(form, "fc (Hz)", "fc", block.params["fc"])
        elif kind == "BP":
            self._fields["f_low"] = self._add_field(form, "f_low (Hz)", "f_low", block.params["f_low"])
            self._fields["f_high"] = self._add_field(form, "f_high (Hz)", "f_high", block.params["f_high"])
            fc_label = QLabel("—")
            fc_label.setObjectName("derivedFc")
            q_label = QLabel("—")
            q_label.setObjectName("derivedQ")
            form.addRow("fc (derived)", fc_label)
            form.addRow("Q (derived)", q_label)
            self._derived_labels["fc"] = fc_label
            self._derived_labels["Q"] = q_label
        elif kind == "AP":
            self._fields["fc"] = self._add_field(form, "fc (Hz)", "fc", block.params["fc"])
            self._fields["Q"] = self._add_field(form, "Q", "Q", block.params["Q"])
        elif kind == "PK":
            self._fields["fc"] = self._add_field(form, "fc (Hz)", "fc", block.params["fc"])
            self._fields["Q"] = self._add_field(form, "Q", "Q", block.params["Q"])
            self._fields["gain_db"] = self._add_field(form, "Gain (dB)", "gain_db", block.params["gain_db"])

    def _add_field(self, form: QFormLayout, label: str, key: str, value: float) -> QLineEdit:
        edit = QLineEdit(f"{value:g}")
        edit.setObjectName(f"field_{key}")
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
        self.chain.update_params(self.block_id, **{key: value})
        self.refresh_live()
        self.params_changed.emit()

    # -- live views (always current, per CONTRACTS.md §13) ----------------------

    def refresh_live(self) -> None:
        block = self.chain.get_block(self.block_id)

        if block.kind == "BP":
            derived = block.filter.derived_params() if block.filter is not None else {}
            self._derived_labels["fc"].setText(f"{derived['fc']:.4g} Hz" if "fc" in derived else "—")
            self._derived_labels["Q"].setText(f"{derived['Q']:.4g}" if "Q" in derived else "—")

        if block.is_valid:
            self.error_label.setVisible(False)
            self.error_label.setText("")
        else:
            self.error_label.setText(block.error)
            self.error_label.setVisible(True)
            self.bode.show_message("This block is invalid. Fix its parameters above to see its response.")
            self.gain.show_message("This block is invalid. Fix its parameters above to see its response.")
            self._current_responses = None
            self._clear_table()
            return

        filt = block.filter
        freq = bode_grid(self.chain.fs)
        ideal_resp = filt.ideal_response(freq)
        q14_resp = filt.q14_response(freq, self.backend) if self.backend is not None else None
        self.bode.plot(ideal_resp, q14_resp)
        self._current_responses = (ideal_resp, q14_resp)

        lin_freq = linear_response_grid(self.chain.fs)
        lin_ideal = filt.ideal_response(lin_freq)
        lin_q14 = filt.q14_response(lin_freq, self.backend) if self.backend is not None else None
        self.gain.plot(lin_ideal, lin_q14)

        self._fill_table(filt)

    def _fill_table(self, filt) -> None:
        if self.backend is None:
            self._clear_table()
            return
        ideal = filt.ideal_coefficients()
        q14 = filt.q14_coefficients(self.backend)
        q14_float = q14.to_float()
        for row, name in enumerate(_COEFFICIENT_NAMES):
            self.table.setItem(row, 0, QTableWidgetItem(f"{getattr(ideal, name):.8f}"))
            self.table.setItem(row, 1, QTableWidgetItem(str(getattr(q14, name))))
            self.table.setItem(row, 2, QTableWidgetItem(f"{getattr(q14_float, name):.8f}"))

    def _clear_table(self) -> None:
        for row in range(len(_COEFFICIENT_NAMES)):
            for col in range(3):
                self.table.setItem(row, col, QTableWidgetItem(""))


class CombinedInspectorPanel(QWidget):
    """The "Combined" tab: series-cascade Bode/gain plots + combined response metric."""

    def __init__(self, chain: FilterChain, backend: NativeBackend | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.chain = chain
        self.backend = backend
        self._current_responses = None  # (ideal, q14) FrequencyResponse pair for the cursor, or None

        outer = QVBoxLayout(self)

        self.status_label = QLabel("")
        self.status_label.setObjectName("combinedStatus")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.bode = BodeWidget()
        outer.addWidget(self.bode, 1)

        self.gain = GainWidget()
        outer.addWidget(self.gain, 1)

        self.cursor = MeasurementCursor(self.bode, self.gain, lambda: self._current_responses)

        self.metrics = _MetricsPanel()
        outer.addWidget(self.metrics)

        self.refresh_live()

    def refresh_live(self) -> None:
        blocks = self.chain.blocks
        valid = self.chain.valid_filters

        if not blocks:
            self.status_label.setText("No filter blocks in the chain. Drag a filter from the palette to add one.")
            self.bode.show_message("No filter blocks in the chain.")
            self.gain.show_message("No filter blocks in the chain.")
            self._current_responses = None
            return
        if not valid:
            self.status_label.setText(
                f"All {len(blocks)} block(s) in the chain are invalid or disabled. "
                "Fix or enable at least one to see the combined response."
            )
            self.bode.show_message("All blocks are invalid or disabled.")
            self.gain.show_message("All blocks are invalid or disabled.")
            self._current_responses = None
            return

        n_excluded = len(blocks) - len(valid)
        if n_excluded:
            n_disabled = sum(1 for b in blocks if not b.enabled)
            n_invalid_enabled = n_excluded - n_disabled
            reasons = "/".join(
                reason
                for reason, count in (("invalid", n_invalid_enabled), ("disabled", n_disabled))
                if count
            )
            self.status_label.setText(
                f"{n_excluded} of {len(blocks)} block(s) are {reasons} and excluded from this combined response."
            )
        else:
            self.status_label.setText("")
        freq = bode_grid(self.chain.fs)
        ideal_resp = self.chain.combined_ideal_response(freq)
        q14_resp = self.chain.combined_q14_response(freq, self.backend) if self.backend is not None else None
        self.bode.plot(ideal_resp, q14_resp)
        self._current_responses = (ideal_resp, q14_resp)

        lin_freq = linear_response_grid(self.chain.fs)
        lin_ideal = self.chain.combined_ideal_response(lin_freq)
        lin_q14 = self.chain.combined_q14_response(lin_freq, self.backend) if self.backend is not None else None
        self.gain.plot(lin_ideal, lin_q14)


class Inspector(QWidget):
    """Tab container: "Combined" + one tab per `FilterChain` block.

    Signals:
        params_changed: a form field edited a block's params through
            `FilterChain.update_params()`; the surrounding window should
            re-render any other view of the model (the canvas) and refresh
            Export's enabled state.
        block_selected(object): the active tab changed (a block id `str`,
            or `None` for "Combined"); the surrounding window should sync
            this selection back onto the canvas.
    """

    params_changed = pyqtSignal()
    block_selected = pyqtSignal(object)

    def __init__(self, chain: FilterChain, backend: NativeBackend | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.chain = chain
        self.backend = backend

        self.tabs = QTabWidget()
        self.tabs.setObjectName("inspectorTabs")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

        self._panels: dict[str | None, QWidget] = {}
        self._response_cache: dict[str | None, ResponseError] = {}
        self._sweep_cache: dict[str, CoefficientSweepError] = {}

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.refresh()

    # -- structural rebuild (add/remove/reorder/fs change) -----------------------

    def refresh(self) -> None:
        """Rebuilds every tab from `chain.blocks`, then re-runs validation."""
        previously_selected = self.chain.selected_id

        self.tabs.blockSignals(True)
        while self.tabs.count():
            self.tabs.removeTab(0)
        self._panels = {}

        combined = CombinedInspectorPanel(self.chain, self.backend)
        self.tabs.addTab(combined, "Combined")
        self._panels[None] = combined

        for position, block in enumerate(self.chain.blocks):
            panel = BlockInspectorPanel(self.chain, block.id, self.backend)
            panel.params_changed.connect(self._on_params_changed)
            self.tabs.addTab(panel, f"FILT{position + 1}: {block.kind}")
            self._panels[block.id] = panel

        target = previously_selected if previously_selected in self._panels else None
        self.chain.select(target)
        self.tabs.setCurrentIndex(list(self._panels.keys()).index(target))
        self.tabs.blockSignals(False)

        self.refresh_validation()

    # -- selection sync -----------------------------------------------------------

    def _on_tab_changed(self, index: int) -> None:
        if not (0 <= index < len(self._panels)):
            return
        block_id = list(self._panels.keys())[index]
        self.chain.select(block_id)
        self._panels[block_id].refresh_live()
        self.block_selected.emit(block_id)

    def select_block(self, block_id: str | None) -> None:
        """External sync (e.g. canvas selection) -> switch to the matching tab."""
        if block_id not in self._panels:
            block_id = None
        self.tabs.setCurrentIndex(list(self._panels.keys()).index(block_id))

    # -- edits from a block's own form -------------------------------------------

    def _on_params_changed(self) -> None:
        self.refresh_validation()
        self.params_changed.emit()

    # -- automatic validation ------------------------------------------------------------

    def _bp_sweep_design_at(self, block: ChainBlock):
        """`coefficient_sweep()` factory for a BP block (CONTRACTS.md §6.3).

        Sweeps the center frequency while holding the block's configured
        bandwidth (`f_high - f_low`) fixed, clamping `f_low`/`f_high` at the
        `[COEFFICIENT_SWEEP_FLOOR_HZ, fc_max(fs)]` domain edges -- identical
        strategy to `export.py::_sweep_design_at`, validated by
        tests/test_error_analysis.py::test_coefficient_sweep_bp_holds_bandwidth_fixed.
        """
        filt = block.filter
        fs = self.chain.fs
        hi = fc_max(fs)
        half_bw = (block.params["f_high"] - block.params["f_low"]) / 2.0

        def design_at(center: float, f=filt, half_bw=half_bw, hi=hi, fs=fs):
            f_low = max(COEFFICIENT_SWEEP_FLOOR_HZ, center - half_bw)
            f_high = min(hi, center + half_bw)
            return type(f)(fs=fs, f_low=f_low, f_high=f_high)

        return design_at

    def refresh_validation(self) -> None:
        """Recomputes every cached response-error / coefficient-sweep metric.

        Runs automatically after every model mutation -- a field edit, an
        add/remove/reorder, a clear/reset, or an `fs` change -- there is no
        manual Validate action anywhere in this UI (CONTRACTS.md §13).
        """
        self._response_cache = {}
        self._sweep_cache = {}

        valid_filters = self.chain.valid_filters
        if valid_filters and self.backend is not None:
            try:
                self._response_cache[None] = combined_response_error(valid_filters, self.backend)
            except ValueError:
                pass

        if self.backend is not None:
            for block in self.chain.blocks:
                if not block.is_valid:
                    continue
                filt = block.filter
                try:
                    self._response_cache[block.id] = response_error(filt, self.backend)
                except ValueError:
                    continue
                try:
                    if block.kind in ("LP", "HP"):
                        self._sweep_cache[block.id] = coefficient_sweep(
                            self.chain.fs, lambda x, f=filt: type(f)(fs=self.chain.fs, fc=x), self.backend
                        )
                    elif block.kind == "AP":
                        q = block.params["Q"]
                        self._sweep_cache[block.id] = coefficient_sweep(
                            self.chain.fs, lambda x, f=filt, q=q: type(f)(fs=self.chain.fs, fc=x, Q=q), self.backend
                        )
                    elif block.kind == "PK":
                        q = block.params["Q"]
                        gain_db = block.params["gain_db"]
                        self._sweep_cache[block.id] = coefficient_sweep(
                            self.chain.fs,
                            lambda x, f=filt, q=q, gain_db=gain_db: type(f)(
                                fs=self.chain.fs, fc=x, Q=q, gain_db=gain_db
                            ),
                            self.backend,
                        )
                    elif block.kind == "BP":
                        self._sweep_cache[block.id] = coefficient_sweep(
                            self.chain.fs, self._bp_sweep_design_at(block), self.backend
                        )
                except ValueError:
                    pass

        self.chain.mark_clean()
        self._render_metrics()

    # -- metrics presentation (cheap: label text/style only) ------------------------

    def _render_metrics(self) -> None:
        combined_panel = self._panels.get(None)
        if combined_panel is not None:
            self._render_combined_metrics(combined_panel)
        for block in self.chain.blocks:
            panel = self._panels.get(block.id)
            if panel is not None:
                self._render_block_metrics(panel, block)

    def _render_combined_metrics(self, panel: CombinedInspectorPanel) -> None:
        if not self.chain.blocks:
            panel.metrics.set_unavailable("Add at least one filter block to compute a combined response.")
        elif not self.chain.valid_filters:
            panel.metrics.set_unavailable(
                f"All {len(self.chain.blocks)} block(s) are invalid or disabled "
                "— cannot compute a combined response."
            )
        elif self.backend is None:
            panel.metrics.set_unavailable("Native backend unavailable — Q14 metrics cannot be computed.")
        else:
            result = self._response_cache.get(None)
            if result is not None:
                panel.metrics.set_result(result, None)

    def _render_block_metrics(self, panel: BlockInspectorPanel, block: ChainBlock) -> None:
        if not block.is_valid:
            panel.metrics.set_unavailable("Block is invalid — cannot compute metrics.")
        elif self.backend is None:
            panel.metrics.set_unavailable("Native backend unavailable — cannot compute metrics.")
        else:
            result = self._response_cache.get(block.id)
            if result is not None:
                sweep = self._sweep_cache.get(block.id)
                panel.metrics.set_result(result, sweep)
