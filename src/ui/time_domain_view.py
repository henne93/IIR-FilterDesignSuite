"""Time-Domain view: second top-level view alongside the Design view (docs/CONCEPT.md §11.7).

Mirrors the Design view's 3-panel `QSplitter` pattern (`ui/app.py`'s
`_build_central_widget`) -- palette / canvas / inspector -- but for the
additive signal generator (`signals.SignalChain`) instead of the filter
chain. `chain` is the same `FilterChain` instance the Design view edits;
this view consumes it read-only (§11.7's last bullet) while owning its own
`SignalChain`.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea, QSplitter, QWidget

from filters import FilterChain
from filters.base import NativeBackend
from signals import SignalChain
from ui.signal_canvas import SignalCanvas
from ui.signal_palette import SignalPalette
from ui.time_domain_inspector import TimeDomainInspector


class TimeDomainView(QSplitter):
    def __init__(
        self, chain: FilterChain, backend: NativeBackend | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setObjectName("timeDomainView")

        self.signal_chain = SignalChain()
        self.palette = SignalPalette()
        self.canvas = SignalCanvas(self.signal_chain)
        self.inspector = TimeDomainInspector(chain, self.signal_chain, backend)

        self.canvas.chain_changed.connect(self.inspector.refresh_live)

        self.addWidget(self.palette)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        self.addWidget(scroll)
        self.addWidget(self.inspector)
        self.setSizes([160, 220, 520])

    def refresh_fs(self) -> None:
        """Re-syncs the read-only fs echo and rebuilds filter-block tabs.

        Called by `MainWindow` after any Design-view filter-chain change
        (fs, add/remove/reorder, or a param edit) -- the same `FilterChain`
        instance backs both views.
        """
        self.inspector.sync_with_filter_chain()
