"""Time-domain Inspector panel (docs/CONCEPT.md §11.6/§11.7).

Rough GUI pass: static controls (duration, read-only fs echo, Q14 full-scale
reference) and a "Combined" tab holding a placeholder `TimeDomainWidget`. No
signal generation or filtering is wired up yet -- per-block tabs (mirroring
`ui/inspector.py`'s Bode tab strip) are a follow-up once a real `SignalChain`
model exists and this panel has something to render per block.
"""

from __future__ import annotations

from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import QFormLayout, QLabel, QLineEdit, QTabWidget, QVBoxLayout, QWidget

from filters import FilterChain
from ui.widgets.time_domain_widget import TimeDomainWidget

DEFAULT_DURATION_MS = 50.0
DEFAULT_Q14_FULL_SCALE = 32767


class TimeDomainInspector(QWidget):
    def __init__(self, chain: FilterChain, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.chain = chain
        self.setObjectName("timeDomainInspector")

        layout = QVBoxLayout(self)

        controls = QFormLayout()
        self.duration_edit = QLineEdit(f"{DEFAULT_DURATION_MS:g}")
        self.duration_edit.setValidator(QDoubleValidator(0.001, 1.0e6, 3))
        self.duration_edit.setMaximumWidth(120)
        controls.addRow("Duration (ms):", self.duration_edit)

        self.fs_label = QLabel(f"{self.chain.fs:g} Hz (from Design)")
        controls.addRow("fs:", self.fs_label)

        self.full_scale_edit = QLineEdit(str(DEFAULT_Q14_FULL_SCALE))
        self.full_scale_edit.setValidator(QIntValidator(1, 32767))
        self.full_scale_edit.setMaximumWidth(120)
        controls.addRow("Q14 full-scale:", self.full_scale_edit)

        layout.addLayout(controls)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("timeDomainInspectorTabs")
        combined_plot = TimeDomainWidget()
        self.tabs.addTab(combined_plot, "Combined")
        layout.addWidget(self.tabs, 1)

    def refresh_fs(self) -> None:
        """Re-syncs the read-only fs echo after the Design view's fs changes."""
        self.fs_label.setText(f"{self.chain.fs:g} Hz (from Design)")
