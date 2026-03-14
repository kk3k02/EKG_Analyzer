from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout


class SamplingRateDialog(QDialog):
    def __init__(self, initial_value: float = 250.0, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sampling rate")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("CSV/TXT file has no explicit time axis. Confirm the sampling rate."))

        self.spin_box = QDoubleSpinBox(self)
        self.spin_box.setRange(0.1, 10000.0)
        self.spin_box.setDecimals(2)
        self.spin_box.setValue(initial_value)
        self.spin_box.setSuffix(" Hz")

        form = QFormLayout()
        form.addRow("Sampling rate:", self.spin_box)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def sampling_rate(self) -> float:
        return float(self.spin_box.value())
