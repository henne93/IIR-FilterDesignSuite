"""Time-domain Inspector panel (docs/CONCEPT.md §11.3-§11.7).

Owns the time-domain-only settings (duration, Q14 full-scale reference) and
a tab strip keyed to the *filter* chain's blocks, mirroring `ui/inspector.py`'s
Bode "Combined" + per-filter-block tabs (§11.6): each tab feeds the same
generated source signal (from `signal_chain`) through either the whole
filter cascade ("Combined") or just that one filter block, so the user can
compare a single block's time-domain effect against the full chain. Signal
blocks themselves have no per-block tab here (§11.6 only speaks of filter
blocks) -- they're edited directly on the canvas tiles instead.

Automatic, always-live (mirrors ui/inspector.py's no-manual-Validate rule,
CONTRACTS.md §13): every open tab recomputes on a signal-chain edit (from
`SignalCanvas.chain_changed`), a filter-chain edit (add/remove/reorder/
param-edit/fs, forwarded from `MainWindow` via `TimeDomainView.refresh_fs()`),
or an edit to this panel's own duration/full-scale fields.
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import QFormLayout, QLabel, QLineEdit, QTabWidget, QVBoxLayout, QWidget

from filters import FilterChain
from filters.base import FilterDesign, NativeBackend
from signals import SignalChain
from time_domain import compute, n_samples_for
from ui.widgets.time_domain_widget import TimeDomainWidget

DEFAULT_DURATION_MS = 50.0
DEFAULT_Q14_FULL_SCALE = 32767

FiltersGetter = Callable[[], "tuple[list[FilterDesign], str | None]"]


class _TimeDomainPanel(QWidget):
    """One tab: status line + time-domain plot for a fixed set of filters (Combined or one block)."""

    def __init__(self, inspector: "TimeDomainInspector", filters_getter: FiltersGetter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._inspector = inspector
        self._filters_getter = filters_getter

        layout = QVBoxLayout(self)
        self.status_label = QLabel("")
        self.status_label.setObjectName("timeDomainStatus")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.plot = TimeDomainWidget()
        layout.addWidget(self.plot, 1)

        self.refresh_live()

    def refresh_live(self) -> None:
        insp = self._inspector
        signal_chain = insp.signal_chain

        if not signal_chain.blocks:
            self.status_label.setText("No signal blocks. Drag a signal block from the palette to add one.")
            self.plot.show_message("No signal blocks in the source signal.")
            return
        if not signal_chain.valid_blocks:
            self.status_label.setText(
                f"All {len(signal_chain.blocks)} signal block(s) are invalid. Fix their parameters above."
            )
            self.plot.show_message("All signal blocks are invalid.")
            return

        try:
            duration_ms = insp.duration_ms()
        except ValueError:
            self.status_label.setText("Duration must be a positive number.")
            self.plot.show_message("Duration must be a positive number.")
            return
        try:
            full_scale = insp.full_scale()
        except ValueError:
            self.status_label.setText("Q14 full-scale must be an integer in [1, 32767].")
            self.plot.show_message("Q14 full-scale must be an integer in [1, 32767].")
            return

        filters_list, filters_error = self._filters_getter()
        if filters_error is not None:
            self.status_label.setText(filters_error)
            self.plot.show_message(filters_error)
            return

        n_excluded = len(signal_chain.blocks) - len(signal_chain.valid_blocks)
        self.status_label.setText(
            f"{n_excluded} signal block(s) are invalid and excluded from the source signal." if n_excluded else ""
        )

        n = n_samples_for(duration_ms, insp.chain.fs)
        source = signal_chain.source_signal(n, insp.chain.fs)
        result = compute(source, filters_list, insp.chain.fs, full_scale, insp.backend)
        self.plot.plot(result.time_ms, result.source, result.ideal, result.q14)


class TimeDomainInspector(QWidget):
    def __init__(
        self,
        chain: FilterChain,
        signal_chain: SignalChain,
        backend: NativeBackend | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.chain = chain
        self.signal_chain = signal_chain
        self.backend = backend
        self.setObjectName("timeDomainInspector")

        layout = QVBoxLayout(self)

        controls = QFormLayout()
        self.duration_edit = QLineEdit(f"{DEFAULT_DURATION_MS:g}")
        self.duration_edit.setValidator(QDoubleValidator(0.001, 1.0e6, 3))
        self.duration_edit.setMaximumWidth(120)
        self.duration_edit.editingFinished.connect(self.refresh_live)
        controls.addRow("Duration (ms):", self.duration_edit)

        self.fs_label = QLabel(f"{self.chain.fs:g} Hz (from Design)")
        controls.addRow("fs:", self.fs_label)

        self.full_scale_edit = QLineEdit(str(DEFAULT_Q14_FULL_SCALE))
        self.full_scale_edit.setValidator(QIntValidator(1, 32767))
        self.full_scale_edit.setMaximumWidth(120)
        self.full_scale_edit.editingFinished.connect(self.refresh_live)
        controls.addRow("Q14 full-scale:", self.full_scale_edit)

        layout.addLayout(controls)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("timeDomainInspectorTabs")
        layout.addWidget(self.tabs, 1)

        self._panels: dict[str | None, _TimeDomainPanel] = {}
        self.refresh_structure()

    # -- settings ------------------------------------------------------------

    def duration_ms(self) -> float:
        value = float(self.duration_edit.text())
        if not value > 0:
            raise ValueError("duration must be > 0")
        return value

    def full_scale(self) -> int:
        return int(self.full_scale_edit.text())

    # -- structure (filter-chain-block tabs, mirrors ui/inspector.py's Inspector.refresh) ------

    def refresh_structure(self) -> None:
        """Rebuilds the Combined + per-filter-block tabs from `chain.blocks`."""
        previous_index = max(self.tabs.currentIndex(), 0)
        self.tabs.blockSignals(True)
        while self.tabs.count():
            self.tabs.removeTab(0)
        self._panels = {}

        combined = _TimeDomainPanel(self, lambda: (self.chain.valid_filters, None))
        self.tabs.addTab(combined, "Combined")
        self._panels[None] = combined

        for position, block in enumerate(self.chain.blocks):
            panel = _TimeDomainPanel(self, self._filters_getter_for(block.id))
            self.tabs.addTab(panel, f"FILT{position + 1}: {block.kind}")
            self._panels[block.id] = panel

        self.tabs.setCurrentIndex(min(previous_index, self.tabs.count() - 1))
        self.tabs.blockSignals(False)

    def _filters_getter_for(self, block_id: str) -> FiltersGetter:
        def getter() -> tuple[list[FilterDesign], str | None]:
            try:
                block = self.chain.get_block(block_id)
            except KeyError:
                return [], "This filter block no longer exists."
            if not block.is_valid:
                return [], f"This filter block is invalid: {block.error}"
            return [block.filter], None

        return getter

    # -- live refresh ----------------------------------------------------------

    def refresh_live(self) -> None:
        """Recomputes every open tab's plot -- signal-chain edit or duration/full-scale edit."""
        for panel in self._panels.values():
            panel.refresh_live()

    def sync_with_filter_chain(self) -> None:
        """Re-syncs the read-only fs echo and rebuilds filter-block tabs.

        Called by `TimeDomainView.refresh_fs()` after any Design-view
        filter-chain change (fs, add/remove/reorder, or a param edit) --
        the same `FilterChain` instance backs both views.
        """
        self.fs_label.setText(f"{self.chain.fs:g} Hz (from Design)")
        self.refresh_structure()
