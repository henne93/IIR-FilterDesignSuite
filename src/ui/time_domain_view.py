"""Time-Domain view: second top-level view alongside the Design view (docs/CONCEPT.md §11.7).

Mirrors the Design view's 3-panel `QSplitter` pattern (`ui/app.py`'s
`_build_central_widget`) -- palette / canvas / inspector -- but for the
signal-generator building blocks instead of the filter chain. Rough GUI pass:
no signal model or filtering logic is wired up yet.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea, QSplitter, QWidget

from filters import FilterChain
from ui.signal_canvas import SignalCanvas
from ui.signal_palette import SignalPalette
from ui.time_domain_inspector import TimeDomainInspector


class TimeDomainView(QSplitter):
    def __init__(self, chain: FilterChain, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setObjectName("timeDomainView")

        self.palette = SignalPalette()
        self.canvas = SignalCanvas()
        self.inspector = TimeDomainInspector(chain)

        self.addWidget(self.palette)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        self.addWidget(scroll)
        self.addWidget(self.inspector)
        self.setSizes([160, 220, 520])

    def refresh_fs(self) -> None:
        self.inspector.refresh_fs()
