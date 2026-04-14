from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSlider,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from app.services.preprocessing import SignalFilterConfig, default_filter_config


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
    filter_config_changed = Signal(object)
    play_requested = Signal()
    pause_requested = Signal()
    stop_requested = Signal()
    playback_speed_changed = Signal(float)
    playback_loop_toggled = Signal(bool)
    playback_position_changed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._filter_config = default_filter_config()
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
        self.sampling_rate_label = QLabel("Nadpisanie czestotliwosci probkowania")
        self.sampling_rate_label.setToolTip(
            "Uzywaj glownie dla danych CSV/TXT bez jawnej kolumny czasu. "
            "WFDB, EDF i DICOM zachowuja czestotliwosc probkowania z pliku zrodlowego."
        )

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

        self.filter_section_checkbox = QCheckBox("Filtrowanie")
        self.filter_section_checkbox.setChecked(False)
        self.filter_section_checkbox.toggled.connect(self._toggle_filter_group_visibility)
        layout.addWidget(self.filter_section_checkbox)

        self.filter_group = self._build_filter_group()
        layout.addWidget(self.filter_group)

        self.playback_group = self._build_playback_group()
        layout.addWidget(self.playback_group)

        leads_group = QGroupBox("Odprowadzenia")
        leads_layout = QVBoxLayout(leads_group)
        self.leads_list = QListWidget(self)
        self.leads_list.setMinimumHeight(180)
        self.leads_list.itemChanged.connect(self._emit_visibility)
        leads_layout.addWidget(self.leads_list)
        layout.addWidget(leads_group)

        mode_group = QGroupBox("Tryb widoku")
        mode_layout = QVBoxLayout(mode_group)
        self.stacked_radio = QRadioButton("Widok wieloodprowadzeniowy")
        self.single_radio = QRadioButton("Skupienie na jednym odprowadzeniu")
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

        layout.addStretch(1)
        self._sync_filter_controls()
        self._toggle_filter_group_visibility(self.filter_section_checkbox.isChecked())
        self.set_playback_enabled(False)

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

    def set_playback_enabled(self, enabled: bool) -> None:
        for widget in (
            self.play_button,
            self.pause_button,
            self.stop_button,
            self.playback_slider,
            self.playback_speed_combo,
            self.loop_checkbox,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self.playback_position_label.setText("0.0 s / 0.0 s")
            self.playback_slider.blockSignals(True)
            self.playback_slider.setValue(0)
            self.playback_slider.blockSignals(False)

    def set_playback_position(self, current_time_sec: float, duration_sec: float) -> None:
        clamped_duration = max(duration_sec, 0.0)
        clamped_time = min(max(current_time_sec, 0.0), clamped_duration)
        self.playback_position_label.setText(f"{clamped_time:.1f} s / {clamped_duration:.1f} s")
        slider_value = 0
        if clamped_duration > 0:
            slider_value = int(round((clamped_time / clamped_duration) * self.playback_slider.maximum()))
        self.playback_slider.blockSignals(True)
        self.playback_slider.setValue(slider_value)
        self.playback_slider.blockSignals(False)

    def set_playback_state(self, state: str) -> None:
        self.playback_state_label.setText(f"Stan: {state}")

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
        self.highpass_cutoff_spin.setToolTip("Czestotliwosc odciecia filtra gornoprzepustowego.")
        self.highpass_cutoff_spin.valueChanged.connect(self._on_filter_control_changed)
        highpass_form = QFormLayout()
        highpass_form.addRow("Odciecie [Hz]", self.highpass_cutoff_spin)
        layout.addLayout(highpass_form)

        self.lowpass_checkbox = QCheckBox("Filtr dolnoprzepustowy")
        self.lowpass_checkbox.setToolTip("Tlumi szum o wysokich czestotliwosciach.")
        self.lowpass_checkbox.toggled.connect(self._on_filter_control_changed)
        layout.addWidget(self.lowpass_checkbox)

        self.lowpass_cutoff_spin = self._make_frequency_spin(DEFAULT=40.0)
        self.lowpass_cutoff_spin.setToolTip("Czestotliwosc odciecia filtra dolnoprzepustowego.")
        self.lowpass_cutoff_spin.valueChanged.connect(self._on_filter_control_changed)
        lowpass_form = QFormLayout()
        lowpass_form.addRow("Odciecie [Hz]", self.lowpass_cutoff_spin)
        layout.addLayout(lowpass_form)

        self.bandpass_checkbox = QCheckBox("Filtr pasmowoprzepustowy")
        self.bandpass_checkbox.setToolTip("Uzywa jednego filtra pasmowego zamiast osobnych etapow gorno- i dolnoprzepustowego.")
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
        self.mains_frequency_combo.currentIndexChanged.connect(self._on_filter_control_changed)
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

    def _build_playback_group(self) -> QGroupBox:
        group = QGroupBox("Sterowanie odtwarzaniem")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        buttons_layout = QHBoxLayout()
        icon_size = QSize(18, 18)
        self.play_button = QPushButton()
        self.pause_button = QPushButton()
        self.stop_button = QPushButton()
        self.play_button.setIcon(self._make_tinted_standard_icon(QStyle.StandardPixmap.SP_MediaPlay, "#FFFFFF"))
        self.pause_button.setIcon(self._make_tinted_standard_icon(QStyle.StandardPixmap.SP_MediaPause, "#FFFFFF"))
        self.stop_button.setIcon(self._make_tinted_standard_icon(QStyle.StandardPixmap.SP_MediaStop, "#FFFFFF"))
        self.play_button.setIconSize(icon_size)
        self.pause_button.setIconSize(icon_size)
        self.stop_button.setIconSize(icon_size)
        self.play_button.setToolTip("Start")
        self.pause_button.setToolTip("Pauza")
        self.stop_button.setToolTip("Stop")
        self.play_button.setAccessibleName("Start")
        self.pause_button.setAccessibleName("Pauza")
        self.stop_button.setAccessibleName("Stop")
        self.play_button.clicked.connect(self.play_requested.emit)
        self.pause_button.clicked.connect(self.pause_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        for button in (self.play_button, self.pause_button, self.stop_button):
            button.setFixedWidth(42)
            button.setStyleSheet(
                "QPushButton { background-color: #263238; border: 1px solid #263238; "
                "border-radius: 6px; padding: 6px; } "
                "QPushButton:disabled { background-color: #CFD8DC; border-color: #CFD8DC; }"
            )
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.stop_button)
        layout.addLayout(buttons_layout)

        self.playback_state_label = QLabel("Stan: Zatrzymane")
        self.playback_position_label = QLabel("0.0 s / 0.0 s")
        layout.addWidget(self.playback_state_label)
        layout.addWidget(self.playback_position_label)

        self.playback_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.playback_slider.setRange(0, 1000)
        self.playback_slider.setSingleStep(1)
        self.playback_slider.sliderMoved.connect(self._emit_playback_position)
        layout.addWidget(self.playback_slider)

        speed_form = QFormLayout()
        self.playback_speed_combo = QComboBox(self)
        for label, value in (("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("4x", 4.0)):
            self.playback_speed_combo.addItem(label, userData=value)
        self.playback_speed_combo.setCurrentIndex(self.playback_speed_combo.findData(1.0))
        self.playback_speed_combo.currentIndexChanged.connect(self._emit_playback_speed)
        speed_form.addRow("Predkosc", self.playback_speed_combo)
        layout.addLayout(speed_form)

        self.loop_checkbox = QCheckBox("Petla")
        self.loop_checkbox.toggled.connect(self.playback_loop_toggled.emit)
        layout.addWidget(self.loop_checkbox)
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
            visibility[int(item.data(Qt.ItemDataRole.UserRole))] = item.checkState() == Qt.CheckState.Checked
        self.lead_visibility_changed.emit(visibility)

    def _emit_view_mode(self) -> None:
        mode = "single" if self.single_radio.isChecked() else "stacked"
        self.active_lead_combo.setEnabled(mode == "single")
        self.view_mode_changed.emit(mode)

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
        self.highpass_cutoff_spin.setEnabled(not bandpass_enabled and self.highpass_checkbox.isChecked())
        self.lowpass_cutoff_spin.setEnabled(not bandpass_enabled and self.lowpass_checkbox.isChecked())
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
        self.mains_frequency_combo.setCurrentIndex(self.mains_frequency_combo.findData(defaults.notch.mains_frequency_hz))
        self.highpass_cutoff_spin.blockSignals(False)
        self.lowpass_cutoff_spin.blockSignals(False)
        self.bandpass_low_spin.blockSignals(False)
        self.bandpass_high_spin.blockSignals(False)
        self.notch_q_spin.blockSignals(False)
        self.mains_frequency_combo.blockSignals(False)
        self._filter_config = defaults
        self._sync_filter_controls()
        self.filter_config_changed.emit(self.filter_config())

    def _emit_playback_speed(self) -> None:
        self.playback_speed_changed.emit(float(self.playback_speed_combo.currentData()))

    def _emit_playback_position(self, slider_value: int) -> None:
        duration_fraction = slider_value / max(self.playback_slider.maximum(), 1)
        self.playback_position_changed.emit(duration_fraction)

    def _make_tinted_standard_icon(self, standard_pixmap: QStyle.StandardPixmap, color_hex: str) -> QIcon:
        base_icon = self.style().standardIcon(standard_pixmap)
        pixmap = base_icon.pixmap(24, 24)
        tinted = pixmap.copy()
        painter = QPainter(tinted)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), QColor(color_hex))
        painter.end()
        return QIcon(tinted)
