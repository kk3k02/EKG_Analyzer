from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ControlsPanel(QWidget):
    load_requested = Signal()
    view_mode_changed = Signal(str)
    active_lead_changed = Signal(int)
    lead_visibility_changed = Signal(object)
    window_preset_selected = Signal(object)
    reset_view_requested = Signal()
    grid_toggled = Signal(bool)
    raw_toggled = Signal(bool)
    filtered_toggled = Signal(bool)
    go_to_start_requested = Signal()
    go_to_end_requested = Signal()
    sampling_rate_changed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.load_button = QPushButton("Wczytaj plik")
        self.load_button.clicked.connect(self.load_requested.emit)
        layout.addWidget(self.load_button)

        self.sampling_rate_spin = QDoubleSpinBox(self)
        self.sampling_rate_spin.setRange(0.1, 10000.0)
        self.sampling_rate_spin.setDecimals(2)
        self.sampling_rate_spin.setSuffix(" Hz")
        self.sampling_rate_spin.setEnabled(False)
        self.sampling_rate_spin.valueChanged.connect(self.sampling_rate_changed.emit)
        self.sampling_rate_label = QLabel("Sampling rate override")
        self.sampling_rate_label.setToolTip(
            "Use this mainly for CSV/TXT data without an explicit time column. "
            "WFDB, EDF and DICOM keep sampling rate from the source file."
        )
        layout.addWidget(self.sampling_rate_label)
        layout.addWidget(self.sampling_rate_spin)

        leads_group = QGroupBox("Odprowadzenia")
        leads_layout = QVBoxLayout(leads_group)
        self.leads_list = QListWidget(self)
        self.leads_list.setMinimumHeight(180)
        self.leads_list.itemChanged.connect(self._emit_visibility)
        leads_layout.addWidget(self.leads_list)
        layout.addWidget(leads_group)

        mode_group = QGroupBox("Tryb widoku")
        mode_layout = QVBoxLayout(mode_group)
        self.stacked_radio = QRadioButton("Stacked multi-lead")
        self.single_radio = QRadioButton("Single lead focus")
        self.stacked_radio.setChecked(True)
        self.mode_buttons = QButtonGroup(self)
        self.mode_buttons.addButton(self.stacked_radio)
        self.mode_buttons.addButton(self.single_radio)
        self.stacked_radio.toggled.connect(self._emit_view_mode)
        self.single_radio.toggled.connect(self._emit_view_mode)
        mode_layout.addWidget(self.stacked_radio)
        mode_layout.addWidget(self.single_radio)
        layout.addWidget(mode_group)

        self.active_lead_combo = QComboBox(self)
        self.active_lead_combo.currentIndexChanged.connect(self.active_lead_changed.emit)
        layout.addWidget(QLabel("Aktywne odprowadzenie"))
        layout.addWidget(self.active_lead_combo)

        preset_group = QGroupBox("Okno czasu")
        preset_layout = QHBoxLayout(preset_group)
        for label, value in (("2 s", 2), ("5 s", 5), ("10 s", 10), ("30 s", 30), ("Caly", None)):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, v=value: self.window_preset_selected.emit(v))
            preset_layout.addWidget(button)
        layout.addWidget(preset_group)

        jump_layout = QHBoxLayout()
        self.start_button = QPushButton("Poczatek")
        self.end_button = QPushButton("Koniec")
        self.start_button.clicked.connect(self.go_to_start_requested.emit)
        self.end_button.clicked.connect(self.go_to_end_requested.emit)
        jump_layout.addWidget(self.start_button)
        jump_layout.addWidget(self.end_button)
        layout.addLayout(jump_layout)

        self.reset_button = QPushButton("Reset widoku")
        self.reset_button.clicked.connect(self.reset_view_requested.emit)
        layout.addWidget(self.reset_button)

        self.grid_checkbox = QCheckBox("Pokaz siatke")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.toggled.connect(self.grid_toggled.emit)
        layout.addWidget(self.grid_checkbox)

        self.filtered_checkbox = QCheckBox("Pokaz sygnal filtrowany do podgladu")
        self.filtered_checkbox.toggled.connect(self.filtered_toggled.emit)
        layout.addWidget(self.filtered_checkbox)

        self.raw_checkbox = QCheckBox("Pokaz surowy sygnal")
        self.raw_checkbox.setChecked(True)
        self.raw_checkbox.toggled.connect(self.raw_toggled.emit)
        layout.addWidget(self.raw_checkbox)

        layout.addStretch(1)

    def set_leads(self, lead_names: list[str]) -> None:
        self.leads_list.blockSignals(True)
        self.active_lead_combo.blockSignals(True)
        self.leads_list.clear()
        self.active_lead_combo.clear()
        for index, name in enumerate(lead_names):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.leads_list.addItem(item)
            self.active_lead_combo.addItem(name, userData=index)
        self.leads_list.blockSignals(False)
        self.active_lead_combo.blockSignals(False)
        self._emit_visibility()

    def set_sampling_rate_controls(self, sampling_rate: float, enabled: bool, tooltip: str) -> None:
        self.sampling_rate_spin.blockSignals(True)
        self.sampling_rate_spin.setValue(sampling_rate)
        self.sampling_rate_spin.setEnabled(enabled)
        self.sampling_rate_spin.setToolTip(tooltip)
        self.sampling_rate_label.setToolTip(tooltip)
        self.sampling_rate_spin.blockSignals(False)

    def _emit_visibility(self) -> None:
        visibility: dict[int, bool] = {}
        for row in range(self.leads_list.count()):
            item = self.leads_list.item(row)
            visibility[int(item.data(Qt.ItemDataRole.UserRole))] = item.checkState() == Qt.CheckState.Checked
        self.lead_visibility_changed.emit(visibility)

    def _emit_view_mode(self) -> None:
        mode = "single" if self.single_radio.isChecked() else "stacked"
        self.active_lead_combo.setEnabled(mode == "single")
        self.view_mode_changed.emit(mode)
