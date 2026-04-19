from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from app.gui.time_utils import format_playback_clock


class PlaybackControlsWidget(QWidget):
    play_requested = Signal()
    pause_requested = Signal()
    stop_requested = Signal()
    step_backward_requested = Signal()
    step_forward_requested = Signal()
    playback_speed_changed = Signal(float)
    playback_loop_toggled = Signal(bool)
    playback_position_changed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.play_button = QPushButton(self)
        self.pause_button = QPushButton(self)
        self.step_backward_button = QPushButton(self)
        self.stop_button = QPushButton(self)
        self.step_forward_button = QPushButton(self)
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.step_backward_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward))
        self.stop_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.step_forward_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward))
        self.play_button.setToolTip("Start")
        self.pause_button.setToolTip("Pauza")
        self.step_backward_button.setToolTip("Wstecz")
        self.stop_button.setToolTip("Stop")
        self.step_forward_button.setToolTip("Do przodu")
        self.play_button.clicked.connect(self.play_requested.emit)
        self.pause_button.clicked.connect(self.pause_requested.emit)
        self.step_backward_button.clicked.connect(self.step_backward_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.step_forward_button.clicked.connect(self.step_forward_requested.emit)
        for button in (
            self.play_button,
            self.pause_button,
            self.step_backward_button,
            self.stop_button,
            self.step_forward_button,
        ):
            button.setFixedWidth(36)
        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addSpacing(12)
        layout.addWidget(self.step_backward_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.step_forward_button)

        self.navigation_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.navigation_slider.setRange(0, 1000)
        self.navigation_slider.setValue(0)
        self.navigation_slider.setEnabled(False)
        self.navigation_slider.setToolTip("Przesun widoczne okno sygnalu.")
        self.navigation_slider.valueChanged.connect(self._emit_playback_position_change)
        layout.addWidget(self.navigation_slider, stretch=1)

        self.playback_position_label = QLabel("00:00 / 00:00", self)
        self.playback_position_label.setMinimumWidth(128)
        self.playback_position_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.playback_position_label)

        self.playback_speed_combo = QComboBox(self)
        for label, value in (("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("4x", 4.0)):
            self.playback_speed_combo.addItem(label, userData=value)
        self.playback_speed_combo.setCurrentIndex(self.playback_speed_combo.findData(1.0))
        self.playback_speed_combo.currentIndexChanged.connect(self._emit_playback_speed_change)

        self.loop_checkbox = QCheckBox("Petla", self)
        self.loop_checkbox.toggled.connect(self.playback_loop_toggled.emit)

        self.playback_settings_button = QToolButton(self)
        self.playback_settings_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.playback_settings_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.playback_settings_button.setToolTip("Ustawienia odtwarzania")
        self.playback_settings_button.setText("Ustawienia")

        self.playback_settings_menu = QMenu(self.playback_settings_button)
        settings_container = QWidget(self.playback_settings_menu)
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(8)
        self.loop_checkbox.setText("")

        playback_speed_row = QHBoxLayout()
        playback_speed_row.setContentsMargins(0, 0, 0, 0)
        playback_speed_row.setSpacing(12)
        playback_speed_label = QLabel("Prędkość", settings_container)
        playback_speed_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        playback_speed_row.addWidget(playback_speed_label)
        playback_speed_row.addStretch(1)
        playback_speed_row.addWidget(self.playback_speed_combo, alignment=Qt.AlignmentFlag.AlignRight)
        settings_layout.addLayout(playback_speed_row)

        loop_row = QHBoxLayout()
        loop_row.setContentsMargins(0, 0, 0, 0)
        loop_row.setSpacing(12)
        loop_label = QLabel("Pętla", settings_container)
        loop_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        loop_row.addWidget(loop_label)
        loop_row.addStretch(1)
        loop_row.addWidget(self.loop_checkbox, alignment=Qt.AlignmentFlag.AlignRight)
        settings_layout.addLayout(loop_row)

        settings_action = QWidgetAction(self.playback_settings_menu)
        settings_action.setDefaultWidget(settings_container)
        self.playback_settings_menu.addAction(settings_action)
        self.playback_settings_button.setMenu(self.playback_settings_menu)
        layout.addWidget(self.playback_settings_button)

        self.set_playback_enabled(False)

    def set_playback_enabled(self, enabled: bool) -> None:
        for widget in (
            self.play_button,
            self.pause_button,
            self.step_backward_button,
            self.stop_button,
            self.step_forward_button,
            self.navigation_slider,
            self.playback_settings_button,
        ):
            widget.setEnabled(enabled)
        self.playback_speed_combo.setEnabled(enabled)
        self.loop_checkbox.setEnabled(enabled)
        if not enabled:
            self.playback_position_label.setText("00:00 / 00:00")
            self.navigation_slider.blockSignals(True)
            self.navigation_slider.setValue(0)
            self.navigation_slider.blockSignals(False)

    def set_playback_position(self, current_time_sec: float, duration_sec: float) -> None:
        clamped_duration = max(float(duration_sec), 0.0)
        clamped_time = min(max(float(current_time_sec), 0.0), clamped_duration)
        self.playback_position_label.setText(
            f"{format_playback_clock(clamped_time)} / {format_playback_clock(clamped_duration)}"
        )
        slider_value = 0
        if clamped_duration > 0.0:
            slider_value = int(round((clamped_time / clamped_duration) * self.navigation_slider.maximum()))
        self.navigation_slider.blockSignals(True)
        self.navigation_slider.setValue(slider_value)
        self.navigation_slider.blockSignals(False)

    def set_playback_state(self, state: str) -> None:
        self.play_button.setProperty("playbackState", state)
        self.play_button.style().unpolish(self.play_button)
        self.play_button.style().polish(self.play_button)

    def _emit_playback_position_change(self, slider_value: int) -> None:
        maximum = max(self.navigation_slider.maximum(), 1)
        self.playback_position_changed.emit(float(slider_value) / float(maximum))

    def _emit_playback_speed_change(self, _index: int) -> None:
        speed = self.playback_speed_combo.currentData()
        if speed is None:
            return
        self.playback_speed_changed.emit(float(speed))
