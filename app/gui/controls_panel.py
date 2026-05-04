from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.services.preprocessing import SignalFilterConfig, default_filter_config


class ControlsPanel(QWidget):
    lead_visibility_changed = Signal(object)
    window_preset_selected = Signal(object)
    reset_view_requested = Signal()
    grid_toggled = Signal(bool)
    raw_toggled = Signal(bool)
    filtered_toggled = Signal(bool)
    sampling_rate_changed = Signal(float)
    filter_config_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._filter_config = default_filter_config()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.sampling_rate_spin = QDoubleSpinBox(self)
        self.sampling_rate_spin.setRange(0.1, 10000.0)
        self.sampling_rate_spin.setDecimals(2)
        self.sampling_rate_spin.setSuffix(" Hz")
        self.sampling_rate_spin.setEnabled(False)
        self.sampling_rate_spin.valueChanged.connect(self.sampling_rate_changed.emit)
        self.sampling_rate_label = QLabel("Nadpisanie czestotliwosci probkowania")
        self.sampling_rate_label.setToolTip(
            "Uzywaj glownie dla danych CSV/TXT bez jawnej kolumny czasu. "
            "WFDB, EDF i DICOM zachowuja czestotliwosc probkowania z pliku zrodlowego."
        )

        preset_group = QGroupBox("Okno czasu")
        preset_layout = QHBoxLayout(preset_group)
        for label, value in (("2 s", 2), ("5 s", 5), ("10 s", 10), ("30 s", 30)):
            button = QPushButton(label)
            button.clicked.connect(
                lambda checked=False, v=value: self.window_preset_selected.emit(v)
            )
            preset_layout.addWidget(button)
        layout.addWidget(preset_group)
        layout.addWidget(self._build_section_separator())

        leads_group = QGroupBox("Odprowadzenia")
        leads_layout = QVBoxLayout(leads_group)
        self.leads_list = QListWidget(self)
        self.leads_list.setMinimumHeight(180)
        self.leads_list.itemChanged.connect(self._emit_visibility)
        leads_layout.addWidget(self.leads_list)
        layout.addWidget(leads_group)
        layout.addWidget(self._build_section_separator())

        layout.addWidget(self.sampling_rate_label)
        layout.addWidget(self.sampling_rate_spin)

        self.reset_button = QPushButton("Reset widoku")
        self.reset_button.clicked.connect(self.reset_view_requested.emit)
        layout.addWidget(self.reset_button)

        self.grid_checkbox = QCheckBox("Pokaz siatke")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.toggled.connect(self.grid_toggled.emit)
        layout.addWidget(self.grid_checkbox)

        self.filtered_checkbox = QCheckBox("Pokaz sygnal przetworzony")
        self.filtered_checkbox.setChecked(True)
        self.filtered_checkbox.toggled.connect(self.filtered_toggled.emit)
        layout.addWidget(self.filtered_checkbox)

        self.raw_checkbox = QCheckBox("Pokaz surowy sygnal jako nakladke")
        self.raw_checkbox.toggled.connect(self.raw_toggled.emit)
        layout.addWidget(self.raw_checkbox)
        layout.addWidget(self._build_section_separator())

        self.filter_section_checkbox = QCheckBox("Filtrowanie")
        self.filter_section_checkbox.setChecked(False)
        self.filter_section_checkbox.toggled.connect(
            self._toggle_filter_group_visibility
        )
        layout.addWidget(self.filter_section_checkbox)

        self.filter_group = self._build_filter_group()
        layout.addWidget(self.filter_group)

        layout.addStretch(1)
        self._sync_filter_controls()
        self._toggle_filter_group_visibility(self.filter_section_checkbox.isChecked())

    def set_leads(self, lead_names: list[str]) -> None:
        self.leads_list.blockSignals(True)
        self.leads_list.clear()
        for index, name in enumerate(lead_names):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.leads_list.addItem(item)
        self.leads_list.blockSignals(False)
        self._emit_visibility()

    def primary_selected_lead_index(self) -> int:
        for row in range(self.leads_list.count()):
            item = self.leads_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                return int(item.data(Qt.ItemDataRole.UserRole))
        return 0

    def set_sampling_rate_controls(
        self, sampling_rate: float, enabled: bool, tooltip: str
    ) -> None:
        self.sampling_rate_spin.blockSignals(True)
        self.sampling_rate_spin.setValue(sampling_rate)
        self.sampling_rate_spin.setEnabled(enabled)
        self.sampling_rate_spin.setToolTip(tooltip)
        self.sampling_rate_label.setToolTip(tooltip)
        self.sampling_rate_spin.blockSignals(False)

    def filter_config(self) -> SignalFilterConfig:
        return SignalFilterConfig(
            dc_removal=self._filter_config.dc_removal,
            highpass=type(self._filter_config.highpass)(
                enabled=self._filter_config.highpass.enabled,
                cutoff=self._filter_config.highpass.cutoff,
            ),
            lowpass=type(self._filter_config.lowpass)(
                enabled=self._filter_config.lowpass.enabled,
                cutoff=self._filter_config.lowpass.cutoff,
            ),
            bandpass=type(self._filter_config.bandpass)(
                enabled=self._filter_config.bandpass.enabled,
                low=self._filter_config.bandpass.low,
                high=self._filter_config.bandpass.high,
            ),
            notch=type(self._filter_config.notch)(
                enabled=self._filter_config.notch.enabled,
                mains_frequency_hz=self._filter_config.notch.mains_frequency_hz,
                quality_factor=self._filter_config.notch.quality_factor,
            ),
        )

    def sync_signal_display_mode(self, filters_active: bool) -> None:
        self.filtered_checkbox.blockSignals(True)
        self.raw_checkbox.blockSignals(True)
        if filters_active:
            self.filtered_checkbox.setChecked(True)
        else:
            self.filtered_checkbox.setChecked(True)
            self.raw_checkbox.setChecked(False)
        self.filtered_checkbox.blockSignals(False)
        self.raw_checkbox.blockSignals(False)
        self.filtered_toggled.emit(self.filtered_checkbox.isChecked())
        self.raw_toggled.emit(self.raw_checkbox.isChecked())

    def set_disease_detection_enabled(self, enabled: bool) -> None:
        self.disease_detection_button.setEnabled(enabled)

    def _build_section_separator(self) -> QFrame:
        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        separator.setFixedHeight(10)
        separator.setStyleSheet(
            "QFrame { color: #8FA3B8; background-color: #8FA3B8; min-height: 3px; max-height: 3px; border: none; }"
        )
        return separator

    def _build_filter_group(self) -> QGroupBox:
        group = QGroupBox("Filtrowanie sygnalu")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.dc_checkbox = QCheckBox("Usun skladowa stala")
        self.dc_checkbox.setToolTip("Odejmuje srednia wartosc z kazdego odprowadzenia.")
        self.dc_checkbox.toggled.connect(self._on_filter_control_changed)
        layout.addWidget(self.dc_checkbox)

        self.highpass_checkbox = QCheckBox("Filtr gornoprzepustowy")
        self.highpass_checkbox.setToolTip("Usuwa wolny dryf linii bazowej.")
        self.highpass_checkbox.toggled.connect(self._on_filter_control_changed)
        layout.addWidget(self.highpass_checkbox)

        self.highpass_cutoff_spin = self._make_frequency_spin(DEFAULT=0.5)
        self.highpass_cutoff_spin.setToolTip(
            "Czestotliwosc odciecia filtra gornoprzepustowego."
        )
        self.highpass_cutoff_spin.valueChanged.connect(self._on_filter_control_changed)
        highpass_form = QFormLayout()
        highpass_form.addRow("Odciecie [Hz]", self.highpass_cutoff_spin)
        layout.addLayout(highpass_form)

        self.lowpass_checkbox = QCheckBox("Filtr dolnoprzepustowy")
        self.lowpass_checkbox.setToolTip("Tlumi szum o wysokich czestotliwosciach.")
        self.lowpass_checkbox.toggled.connect(self._on_filter_control_changed)
        layout.addWidget(self.lowpass_checkbox)

        self.lowpass_cutoff_spin = self._make_frequency_spin(DEFAULT=40.0)
        self.lowpass_cutoff_spin.setToolTip(
            "Czestotliwosc odciecia filtra dolnoprzepustowego."
        )
        self.lowpass_cutoff_spin.valueChanged.connect(self._on_filter_control_changed)
        lowpass_form = QFormLayout()
        lowpass_form.addRow("Odciecie [Hz]", self.lowpass_cutoff_spin)
        layout.addLayout(lowpass_form)

        self.bandpass_checkbox = QCheckBox("Filtr pasmowoprzepustowy")
        self.bandpass_checkbox.setToolTip(
            "Uzywa jednego filtra pasmowego zamiast osobnych etapow gorno- i dolnoprzepustowego."
        )
        self.bandpass_checkbox.toggled.connect(self._on_bandpass_toggled)
        layout.addWidget(self.bandpass_checkbox)

        self.bandpass_low_spin = self._make_frequency_spin(DEFAULT=0.5)
        self.bandpass_high_spin = self._make_frequency_spin(DEFAULT=40.0)
        self.bandpass_low_spin.valueChanged.connect(self._on_filter_control_changed)
        self.bandpass_high_spin.valueChanged.connect(self._on_filter_control_changed)
        bandpass_form = QFormLayout()
        bandpass_form.addRow("Dolne odciecie [Hz]", self.bandpass_low_spin)
        bandpass_form.addRow("Gorne odciecie [Hz]", self.bandpass_high_spin)
        layout.addLayout(bandpass_form)

        self.notch_checkbox = QCheckBox("Filtr notch")
        self.notch_checkbox.setToolTip("Usuwa zaklocenia sieciowe 50/60 Hz.")
        self.notch_checkbox.toggled.connect(self._on_filter_control_changed)
        layout.addWidget(self.notch_checkbox)

        self.mains_frequency_combo = QComboBox(self)
        self.mains_frequency_combo.addItem("50 Hz", userData=50.0)
        self.mains_frequency_combo.addItem("60 Hz", userData=60.0)
        self.mains_frequency_combo.currentIndexChanged.connect(
            self._on_filter_control_changed
        )
        self.notch_q_spin = QDoubleSpinBox(self)
        self.notch_q_spin.setRange(1.0, 100.0)
        self.notch_q_spin.setDecimals(1)
        self.notch_q_spin.setSingleStep(1.0)
        self.notch_q_spin.setValue(30.0)
        self.notch_q_spin.valueChanged.connect(self._on_filter_control_changed)
        notch_form = QFormLayout()
        notch_form.addRow("Czestotliwosc sieci", self.mains_frequency_combo)
        notch_form.addRow("Wspolczynnik Q", self.notch_q_spin)
        layout.addLayout(notch_form)

        self.reset_filters_button = QPushButton("Reset filtrow")
        self.reset_filters_button.clicked.connect(self._reset_filters)
        layout.addWidget(self.reset_filters_button)
        return group

    def _make_frequency_spin(self, *, DEFAULT: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(0.01, 1000.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setValue(DEFAULT)
        spin.setSuffix(" Hz")
        return spin

    def _emit_visibility(self) -> None:
        visibility: dict[int, bool] = {}
        for row in range(self.leads_list.count()):
            item = self.leads_list.item(row)
            visibility[int(item.data(Qt.ItemDataRole.UserRole))] = (
                item.checkState() == Qt.CheckState.Checked
            )
        self.lead_visibility_changed.emit(visibility)

    def _toggle_filter_group_visibility(self, visible: bool) -> None:
        self.filter_group.setVisible(visible)

    def _on_bandpass_toggled(self, checked: bool) -> None:
        if checked:
            self.highpass_checkbox.blockSignals(True)
            self.lowpass_checkbox.blockSignals(True)
            self.highpass_checkbox.setChecked(False)
            self.lowpass_checkbox.setChecked(False)
            self.highpass_checkbox.blockSignals(False)
            self.lowpass_checkbox.blockSignals(False)
        self._sync_filter_controls()
        self._on_filter_control_changed()

    def _on_filter_control_changed(self, *_args: object) -> None:
        self._filter_config = SignalFilterConfig(
            dc_removal=self.dc_checkbox.isChecked(),
            highpass=type(self._filter_config.highpass)(
                enabled=self.highpass_checkbox.isChecked(),
                cutoff=self.highpass_cutoff_spin.value(),
            ),
            lowpass=type(self._filter_config.lowpass)(
                enabled=self.lowpass_checkbox.isChecked(),
                cutoff=self.lowpass_cutoff_spin.value(),
            ),
            bandpass=type(self._filter_config.bandpass)(
                enabled=self.bandpass_checkbox.isChecked(),
                low=self.bandpass_low_spin.value(),
                high=self.bandpass_high_spin.value(),
            ),
            notch=type(self._filter_config.notch)(
                enabled=self.notch_checkbox.isChecked(),
                mains_frequency_hz=float(self.mains_frequency_combo.currentData()),
                quality_factor=self.notch_q_spin.value(),
            ),
        )
        self._sync_filter_controls()
        self.filter_config_changed.emit(self.filter_config())

    def _sync_filter_controls(self) -> None:
        bandpass_enabled = self.bandpass_checkbox.isChecked()
        self.highpass_checkbox.setEnabled(not bandpass_enabled)
        self.lowpass_checkbox.setEnabled(not bandpass_enabled)
        self.highpass_cutoff_spin.setEnabled(
            not bandpass_enabled and self.highpass_checkbox.isChecked()
        )
        self.lowpass_cutoff_spin.setEnabled(
            not bandpass_enabled and self.lowpass_checkbox.isChecked()
        )
        self.bandpass_low_spin.setEnabled(bandpass_enabled)
        self.bandpass_high_spin.setEnabled(bandpass_enabled)
        self.mains_frequency_combo.setEnabled(self.notch_checkbox.isChecked())
        self.notch_q_spin.setEnabled(self.notch_checkbox.isChecked())

    def _reset_filters(self) -> None:
        defaults = default_filter_config()
        widgets = (
            (self.dc_checkbox, defaults.dc_removal),
            (self.highpass_checkbox, defaults.highpass.enabled),
            (self.lowpass_checkbox, defaults.lowpass.enabled),
            (self.bandpass_checkbox, defaults.bandpass.enabled),
            (self.notch_checkbox, defaults.notch.enabled),
        )
        for widget, value in widgets:
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        self.highpass_cutoff_spin.blockSignals(True)
        self.lowpass_cutoff_spin.blockSignals(True)
        self.bandpass_low_spin.blockSignals(True)
        self.bandpass_high_spin.blockSignals(True)
        self.notch_q_spin.blockSignals(True)
        self.mains_frequency_combo.blockSignals(True)
        self.highpass_cutoff_spin.setValue(defaults.highpass.cutoff)
        self.lowpass_cutoff_spin.setValue(defaults.lowpass.cutoff)
        self.bandpass_low_spin.setValue(defaults.bandpass.low)
        self.bandpass_high_spin.setValue(defaults.bandpass.high)
        self.notch_q_spin.setValue(defaults.notch.quality_factor)
        self.mains_frequency_combo.setCurrentIndex(
            self.mains_frequency_combo.findData(defaults.notch.mains_frequency_hz)
        )
        self.highpass_cutoff_spin.blockSignals(False)
        self.lowpass_cutoff_spin.blockSignals(False)
        self.bandpass_low_spin.blockSignals(False)
        self.bandpass_high_spin.blockSignals(False)
        self.notch_q_spin.blockSignals(False)
        self.mains_frequency_combo.blockSignals(False)
        self._filter_config = defaults
        self._sync_filter_controls()
        self.filter_config_changed.emit(self.filter_config())
