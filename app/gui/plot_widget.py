from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from app.gui.playback_controls import PlaybackControlsWidget
from app.models.ecg_record import ECGRecord
from app.services.frequency_overview import (
    DEFAULT_MAX_FREQUENCY_HZ,
    FrequencyOverviewResult,
    compute_frequency_overview,
)
from app.services.selection_stats import compute_selection_stats


@dataclass(slots=True)
class CursorInfo:
    time_value: float
    sample_index: int
    amplitude: float
    lead_name: str


class ECGSelectionViewBox(pg.ViewBox):
    def __init__(self, drag_callback, *args, **kwargs) -> None:
        kwargs.setdefault("enableMenu", False)
        super().__init__(*args, **kwargs)
        self._drag_callback = drag_callback

    def mouseDragEvent(self, ev, axis=None) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_callback(ev)
            return
        ev.ignore()


class ECGPlotWidget(QWidget):
    """Interactive Stage 1 ECG viewer built around pyqtgraph.
    Selection statistics emitted from this widget refer only to the active lead
    in single-lead mode or to the lead nearest the interaction point in stacked
    mode. The widget does not provide clinical annotations or multi-lead
    diagnostic measurements yet..
    """

    load_requested = Signal()
    cursor_changed = Signal(object)
    selection_changed = Signal(object)
    selection_context_menu_requested = Signal()
    play_requested = Signal()
    pause_requested = Signal()
    stop_requested = Signal()
    step_backward_requested = Signal()
    step_forward_requested = Signal()
    playback_speed_changed = Signal(float)
    playback_loop_toggled = Signal(bool)
    playback_position_changed = Signal(float)
    visible_time_range_changed = Signal()

    annotations_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 20, 0, 0)
        layout.setSpacing(10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        pg.setConfigOptions(antialias=False)

        self._main_view_box = ECGSelectionViewBox(self._on_plot_drag_selection)
        self.main_plot = pg.PlotWidget(viewBox=self._main_view_box)
        self.main_plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.main_plot.setBackground("w")
        self.main_plot.setMouseEnabled(x=False, y=False)
        self.main_plot.showGrid(x=True, y=True, alpha=0.25)
        self.main_plot.setLabel("bottom", "Czas", units="s")
        self.main_plot.setLabel("left", "Amplituda")
        self.main_plot.addLegend(offset=(10, 10))
        self.main_plot.plotItem.vb.sigXRangeChanged.connect(
            self._on_visible_time_range_changed
        )

        self.plot_area = QWidget(self)
        plot_area_layout = QStackedLayout(self.plot_area)
        plot_area_layout.setContentsMargins(0, 0, 0, 0)
        plot_area_layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        plot_area_layout.addWidget(self.main_plot)

        self.empty_state_overlay = QWidget(self.plot_area)
        self.empty_state_overlay.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True
        )
        self.empty_state_overlay.setStyleSheet("background: transparent;")
        empty_state_layout = QVBoxLayout(self.empty_state_overlay)
        empty_state_layout.setContentsMargins(24, 24, 24, 24)
        empty_state_layout.setSpacing(12)
        empty_state_layout.addStretch(1)

        self.empty_state_button = QPushButton("Wczytaj plik", self.empty_state_overlay)
        self.empty_state_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.empty_state_button.setMinimumWidth(180)
        self.empty_state_button.setMinimumHeight(48)
        self.empty_state_button.setStyleSheet(
            "QPushButton { background-color: rgba(255, 255, 255, 0.96); border: 1px solid #CFD8DC; "
            "border-radius: 10px; padding: 10px 18px; font-size: 15px; font-weight: 600; color: #263238; } "
            "QPushButton:hover { background-color: rgba(255, 255, 255, 1.0); border-color: #90A4AE; } "
            "QPushButton:pressed { background-color: rgba(236, 239, 241, 1.0); }"
        )
        self.empty_state_button.clicked.connect(self.load_requested.emit)
        empty_state_layout.addWidget(
            self.empty_state_button, alignment=Qt.AlignmentFlag.AlignCenter
        )
        empty_state_layout.addStretch(1)
        plot_area_layout.addWidget(self.empty_state_overlay)

        self.playback_controls = PlaybackControlsWidget(self)
        self.playback_controls.play_requested.connect(self.play_requested.emit)
        self.playback_controls.pause_requested.connect(self.pause_requested.emit)
        self.playback_controls.stop_requested.connect(self.stop_requested.emit)
        self.playback_controls.step_backward_requested.connect(
            self.step_backward_requested.emit
        )
        self.playback_controls.step_forward_requested.connect(
            self.step_forward_requested.emit
        )
        self.playback_controls.playback_speed_changed.connect(
            self.playback_speed_changed.emit
        )
        self.playback_controls.playback_loop_toggled.connect(
            self.playback_loop_toggled.emit
        )
        self.playback_controls.playback_position_changed.connect(
            self.playback_position_changed.emit
        )

        overview_container = QWidget(self)
        overview_layout = QVBoxLayout(overview_container)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.setSpacing(4)

        overview_header = QHBoxLayout()
        overview_header.setContentsMargins(0, 0, 0, 0)
        self.overview_title = QLabel("Przeglad czestotliwosci (FFT)", self)
        self.overview_title.setStyleSheet("font-weight: 600;")
        self.log_scale_checkbox = QCheckBox("Skala log", self)
        self.log_scale_checkbox.toggled.connect(self._on_overview_controls_changed)
        overview_header.addWidget(self.overview_title)
        overview_header.addStretch(1)
        overview_header.addWidget(self.log_scale_checkbox)
        overview_layout.addLayout(overview_header)

        self.overview_plot = pg.PlotWidget()
        self.overview_plot.setMinimumHeight(120)
        self.overview_plot.setMaximumHeight(200)
        self.overview_plot.setBackground("w")
        self.overview_plot.setMouseEnabled(x=False, y=False)
        self.overview_plot.showGrid(x=True, y=True, alpha=0.15)
        self.overview_plot.setLabel("bottom", "Czestotliwosc", units="Hz")
        self.overview_plot.setLabel("left", "Amplituda")

        self.overview_empty_label = QLabel("Wczytaj plik, aby zobaczyc analize.", self)
        self.overview_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overview_empty_label.setStyleSheet("color: #666666;")

        self.overview_stack = QStackedLayout()
        self.overview_stack.addWidget(self.overview_plot)
        self.overview_stack.addWidget(self.overview_empty_label)
        overview_layout.addLayout(self.overview_stack)

        layout.addWidget(self.plot_area, stretch=2)
        layout.addWidget(self.playback_controls, stretch=0)
        layout.addWidget(overview_container, stretch=1)

        self._record: ECGRecord | None = None
        self._preview_signal: np.ndarray | None = None
        self._filtering_active = False
        self._raw_visible = False
        self._filtered_visible = True
        self._show_grid = True
        self._view_mode = "stacked"
        self._active_lead = 0
        self._lead_visibility: dict[int, bool] = {}
        self._window_seconds: int | None = 5
        self._monitor_curves: list[pg.PlotDataItem] = []
        self._annotation_items: list[pg.TextItem] = []
        self._overview_plot_item: pg.PlotDataItem | None = None
        self._frequency_markers: list[pg.InfiniteLine] = []
        self._overview_mode = "frequency"
        self._max_frequency_hz = DEFAULT_MAX_FREQUENCY_HZ
        self._last_frequency_cache_key: tuple[int, int, int, bool, float] | None = None
        self._last_visible_time_range_key: tuple[float, float] | None = None

        self._current_playback_time = 0.0
        self._cursor_pos = 0.0

        self.cursor_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#D32F2F", width=4)
        )
        self.main_plot.addItem(self.cursor_line, ignoreBounds=True)

        self.selection_region = pg.LinearRegionItem(
            values=(0, 1), movable=True, brush=(200, 30, 30, 40)
        )
        self.selection_region.setZValue(10)
        self.selection_region.sigRegionChanged.connect(self._emit_selection_stats)
        self.main_plot.addItem(self.selection_region, ignoreBounds=True)
        self.selection_region.hide()

        self._mouse_proxy = pg.SignalProxy(
            self.main_plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )
        self._main_plot_mouse_proxy = pg.SignalProxy(
            self.main_plot.scene().sigMouseClicked,
            rateLimit=30,
            slot=self._on_mouse_clicked,
        )

        self._overview_update_timer = QTimer(self)
        self._overview_update_timer.setSingleShot(True)
        self._overview_update_timer.setInterval(120)
        self._overview_update_timer.timeout.connect(self.update_frequency_overview_plot)

        self._render()

    # ------------------------------------------------------------------ #
    #  Public API — unchanged                                             #
    # ------------------------------------------------------------------ #

    def set_record(
        self,
        record: ECGRecord | None,
        preview_signal: np.ndarray | None = None,
        *,
        filtering_active: bool = False,
    ) -> None:
        self._record = record
        self._preview_signal = preview_signal
        self._filtering_active = filtering_active
        self._lead_visibility = (
            {index: True for index in range(record.n_leads)} if record else {}
        )
        self._cursor_pos = float(record.time_axis[0]) if record else 0.0
        self._last_frequency_cache_key = None
        self._last_visible_time_range_key = None
        self._render()

    def set_playback_enabled(self, enabled: bool) -> None:
        self.playback_controls.set_playback_enabled(enabled)

    def set_playback_position(
        self, current_time_sec: float, duration_sec: float
    ) -> None:
        self.playback_controls.set_playback_position(current_time_sec, duration_sec)

    def set_playback_state(self, state: str) -> None:
        self.playback_controls.set_playback_state(state)

    def set_raw_visible(self, visible: bool) -> None:
        self._raw_visible = visible
        self._render()

    def set_filtered_visible(self, visible: bool) -> None:
        self._filtered_visible = visible
        self._render()

    def set_grid_visible(self, visible: bool) -> None:
        self._show_grid = visible
        self.main_plot.showGrid(x=visible, y=visible, alpha=0.25)

    def set_view_mode(self, mode: str) -> None:
        self._view_mode = mode
        self._render()

    def set_active_lead(self, lead_index: int) -> None:
        self._active_lead = max(0, lead_index)
        self._render()

    def set_lead_visibility(self, visibility: dict[int, bool]) -> None:
        self._lead_visibility = visibility
        self._render()

    def set_window_seconds(self, seconds: int | None) -> None:
        self._window_seconds = seconds
        self._render()

    def set_visible_time_window(self, start_time: float, window_seconds: float) -> None:
        if self._record is None:
            return

        win_new = window_seconds if window_seconds > 0 else 5.0
        overlap = 1.0
        record_start = float(self._record.time_axis[0])
        playback_time = max(float(start_time), 0.0)
        absolute_cursor_time = record_start + playback_time

        self._current_playback_time = absolute_cursor_time
        self._cursor_pos = absolute_cursor_time

        page_idx = int(playback_time / win_new)
        page_start_raw = record_start + (page_idx * win_new)

        view_start = max(record_start, page_start_raw - overlap)
        view_end = view_start + win_new + (overlap if page_idx > 0 else 0)

        self.main_plot.setXRange(view_start, view_end, padding=0.0)
        self.cursor_line.setPos(absolute_cursor_time)

        self._update_revealed_data(view_start, absolute_cursor_time)

    def _update_revealed_data(self, view_start: float, cursor_pos: float) -> None:
        if self._record is None:
            return

        start_idx, end_idx = self._time_range_to_indices(view_start, cursor_pos)
        if end_idx <= start_idx:
            return

        x_data = self._record.time_axis[start_idx:end_idx]
        display_signal = self._display_signal()
        visible_indices = self._visible_indices()
        offsets = self._compute_offsets(display_signal, visible_indices)

        for i, lead_idx in enumerate(visible_indices):
            if i >= len(self._monitor_curves):
                break
            y_data = display_signal[start_idx:end_idx, lead_idx] + offsets.get(
                lead_idx, 0.0
            )
            self._monitor_curves[i].setData(x_data, y_data)

        for item in self._annotation_items:
            pos = item.pos()
            if view_start <= pos.x() <= cursor_pos:
                item.show()
            else:
                item.hide()

    def _render(self) -> None:
        self.main_plot.clear()
        self._monitor_curves.clear()
        self._annotation_items.clear()
        self.main_plot.addItem(self.cursor_line, ignoreBounds=True)
        self.main_plot.addItem(self.selection_region, ignoreBounds=True)

        if self._record is None:
            self.selection_region.hide()
            self.empty_state_overlay.show()
            self.empty_state_overlay.raise_()
            self._show_frequency_overview_message("Wczytaj plik, aby zobaczyc analize.")
            return

        self.empty_state_overlay.hide()

        display_signal = self._display_signal()
        visible_indices = self._visible_indices()
        offsets = self._compute_offsets(display_signal, visible_indices)

        for lead_idx in visible_indices:
            curve = pg.PlotDataItem(pen=pg.mkPen(color="#D32F2F", width=2.0))
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            curve.setSkipFiniteCheck(True)
            self.main_plot.addItem(curve)
            self._monitor_curves.append(curve)

            if self._view_mode == "stacked":
                label = pg.TextItem(
                    text=self._record.lead_names[lead_idx],
                    color="#D32F2F",
                    anchor=(0, 0.5),
                )
                label.setPos(
                    float(self._record.time_axis[0]), offsets.get(lead_idx, 0.0)
                )
                self.main_plot.addItem(label)

        if self._record.annotations:
            lead_idx = self._active_lead if self._view_mode == "single" else 0
            offset = offsets.get(lead_idx, 0.0)

            for ann in self._record.annotations:
                sample_index = ann.get("sample", 0)
                if 0 <= sample_index < len(self._record.time_axis):
                    # Y position is always computed through the shared helper
                    # so every label — whether loaded from file or added
                    # interactively — sits at exactly the same height above
                    # its QRS peak.
                    y_pos = self._annotation_y_pos(sample_index)
                    txt = pg.TextItem(
                        text=ann.get("symbol", "?"), color="#000000", anchor=(0.5, 1)
                    )
                    txt.setPos(float(self._record.time_axis[sample_index]), y_pos)
                    txt.hide()
                    self.main_plot.addItem(txt)
                    self._annotation_items.append(txt)

        # Fill curves with data from the currently visible time range so that
        # a _render() triggered by annotation add/remove doesn't leave the
        # plot blank.  When playback is active the next _update_revealed_data
        # call will overwrite this anyway.
        view_start, view_end = self.main_plot.viewRange()[0]
        bounded_start = max(float(self._record.time_axis[0]), view_start)
        bounded_end = min(float(self._record.time_axis[-1]), view_end)
        if bounded_end > bounded_start:
            start_idx, end_idx = self._time_range_to_indices(bounded_start, bounded_end)
            x_data = self._record.time_axis[start_idx:end_idx]
            for i, lead_idx in enumerate(visible_indices):
                if i >= len(self._monitor_curves):
                    break
                y_data = display_signal[start_idx:end_idx, lead_idx] + offsets.get(
                    lead_idx, 0.0
                )
                self._monitor_curves[i].setData(x_data, y_data)

            # Show annotation labels that fall within the visible range
            for item in self._annotation_items:
                pos = item.pos()
                if bounded_start <= pos.x() <= bounded_end:
                    item.show()

        self._schedule_frequency_overview_update(immediate=True)

    def update_frequency_overview_plot(self) -> None:
        if self._record is None:
            return

        start, end = self.main_plot.viewRange()[0]
        start_idx, end_idx = self._time_range_to_indices(start, end)
        lead_idx = self._get_frequency_overview_lead_index()
        if lead_idx is None:
            return

        cache_key = (
            start_idx,
            end_idx,
            lead_idx,
            self.log_scale_checkbox.isChecked(),
            self._max_frequency_hz,
        )
        if cache_key == self._last_frequency_cache_key:
            return

        display_signal = self._display_signal()
        signal_segment = display_signal[start_idx:end_idx, lead_idx]
        if signal_segment.size < 16:
            self._last_frequency_cache_key = None
            self._show_frequency_overview_message("Zbyt malo danych do FFT.")
            return

        result = compute_frequency_overview(
            signal_segment,
            self._record.sampling_rate,
            max_frequency_hz=self._max_frequency_hz,
        )
        if result.message is not None:
            self._last_frequency_cache_key = None
            self._show_frequency_overview_message(result.message)
            return

        self._last_frequency_cache_key = cache_key
        self._render_frequency_overview(result, self._record.lead_names[lead_idx])

    def _render_frequency_overview(
        self, result: FrequencyOverviewResult, lead_name: str
    ) -> None:
        self.overview_plot.clear()
        self.overview_plot.showGrid(x=True, y=True, alpha=0.15)
        self.overview_plot.setLabel("left", result.y_label)
        self.overview_plot.setTitle(f"{lead_name} (Analiza okna)")

        values = np.asarray(result.values, dtype=float)
        if self.log_scale_checkbox.isChecked():
            values = np.maximum(values, np.finfo(float).tiny)
            self.overview_plot.setLogMode(x=False, y=True)
        else:
            self.overview_plot.setLogMode(x=False, y=False)

        frequencies = np.asarray(result.frequencies_hz, dtype=float)
        if frequencies.size > 1:
            bar_width = max(float(np.min(np.diff(frequencies))) * 0.85, 1e-6)
        else:
            bar_width = max(self._max_frequency_hz * 0.02, 0.1)
        bars = pg.BarGraphItem(
            x=frequencies,
            height=values,
            width=bar_width,
            brush=pg.mkBrush("#00429d"),
            pen=pg.mkPen("#00429d"),
        )
        self.overview_plot.addItem(bars)
        self.overview_plot.setXRange(0.0, self._max_frequency_hz, padding=0.01)
        self.overview_stack.setCurrentWidget(self.overview_plot)

    def _compute_offsets(
        self, signal: np.ndarray, visible_indices: list[int]
    ) -> dict[int, float]:
        if self._view_mode == "single":
            return {self._active_lead: 0.0}
        return {idx: -i * 3.0 for i, idx in enumerate(visible_indices)}

    def _get_frequency_overview_lead_index(self) -> int | None:
        if self._record is None:
            return None
        if self._view_mode == "single":
            return self._active_lead
        vis = self._visible_indices()
        return vis[0] if vis else None

    def _show_frequency_overview_message(self, message: str) -> None:
        self.overview_empty_label.setText(message)
        self.overview_stack.setCurrentWidget(self.overview_empty_label)

    def _schedule_frequency_overview_update(self, *, immediate: bool = False) -> None:
        self._last_frequency_cache_key = None
        if immediate:
            self._overview_update_timer.stop()
            self.update_frequency_overview_plot()
            return
        self._overview_update_timer.start()

    def _on_visible_time_range_changed(self, *_args) -> None:
        current_range = self.visible_time_range()
        if current_range is None:
            self._last_visible_time_range_key = None
            return
        if current_range == self._last_visible_time_range_key:
            return
        self._last_visible_time_range_key = current_range
        self.visible_time_range_changed.emit()
        self._schedule_frequency_overview_update(immediate=True)

    def _display_signal(self) -> np.ndarray:
        if self._filtering_active and self._preview_signal is not None:
            return self._preview_signal
        return self._record.signal

    def _visible_indices(self) -> list[int]:
        if self._record is None:
            return []
        if self._view_mode == "single":
            return [self._active_lead]
        return [
            i for i in range(self._record.n_leads) if self._lead_visibility.get(i, True)
        ]

    def _time_range_to_indices(
        self, start_time: float, end_time: float
    ) -> tuple[int, int]:
        if self._record is None:
            return 0, 0
        time_axis = self._record.time_axis
        start_idx = int(np.searchsorted(time_axis, start_time, side="left"))
        end_idx = int(np.searchsorted(time_axis, end_time, side="right"))
        start_idx = max(0, min(start_idx, self._record.n_samples))
        end_idx = max(start_idx, min(end_idx, self._record.n_samples))
        return start_idx, end_idx

    def _on_overview_controls_changed(self, _checked: bool) -> None:
        self._schedule_frequency_overview_update(immediate=True)

    def _on_mouse_moved(self, event: tuple[object]) -> None:
        if self._record is None:
            return
        pos = event[0]
        if not self.main_plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
        self._emit_cursor_info_at_pos(float(mouse_point.x()))

    def _emit_cursor_info_at_pos(self, x_value: float) -> None:
        if self._record is None:
            return
        idx = int(
            np.clip(
                np.searchsorted(self._record.time_axis, x_value),
                0,
                self._record.n_samples - 1,
            )
        )
        lead_idx = self._active_lead if self._view_mode == "single" else 0
        if x_value < self._record.time_axis[0] or x_value > self._record.time_axis[-1]:
            return
        self.cursor_changed.emit(
            CursorInfo(
                time_value=float(self._record.time_axis[idx]),
                sample_index=idx,
                amplitude=float(self._record.signal[idx, lead_idx]),
                lead_name=self._record.lead_names[lead_idx],
            )
        )

    # ------------------------------------------------------------------ #
    #  Annotation helpers                                                  #
    # ------------------------------------------------------------------ #

    def _annotation_y_pos(self, sample_index: int) -> float:
        """Return the standardised Y coordinate for an annotation marker.

        Position = signal value at *sample_index* (QRS peak)
                  + lead offset (stacked mode)
                  + margin proportional to the lead's amplitude range.

        This guarantees every label sits at the same visual distance above
        its QRS regardless of zoom level or signal amplitude.
        """
        if self._record is None:
            return 0.0
        display_signal = self._display_signal()
        lead_idx = self._active_lead if self._view_mode == "single" else 0
        visible_indices = self._visible_indices()
        offsets = self._compute_offsets(display_signal, visible_indices)
        offset = offsets.get(lead_idx, 0.0)
        lead_signal = display_signal[:, lead_idx]
        sig_range = float(np.ptp(lead_signal)) if lead_signal.size > 0 else 1.0
        ann_margin = sig_range * 0.08
        return float(display_signal[sample_index, lead_idx]) + offset + ann_margin

    def _snap_to_local_peak(self, sample_index: int) -> int:
        """Return the index of the local signal maximum in a ±50 ms window
        around *sample_index*.  Ensures the annotation sits on the actual
        QRS peak rather than the arbitrary click point within the complex.
        """
        if self._record is None:
            return sample_index
        half_win = max(1, int(self._record.sampling_rate * 0.05))
        lo = max(0, sample_index - half_win)
        hi = min(self._record.n_samples - 1, sample_index + half_win)
        lead_idx = self._active_lead if self._view_mode == "single" else 0
        display_signal = self._display_signal()
        window = display_signal[lo : hi + 1, lead_idx]
        return int(lo + int(np.argmax(window)))

    def _annotation_near_time(self, x_time: float) -> dict | None:
        """Return the annotation dict closest to *x_time* if within tolerance,
        otherwise None.  Tolerance is expressed as a fraction of the visible
        time window so it scales naturally with zoom level.
        """
        if self._record is None or not self._record.annotations:
            return None

        view_start, view_end = self.main_plot.viewRange()[0]
        view_width = max(float(view_end - view_start), 1e-6)
        # Click within 2 % of the current view width counts as "on the annotation"
        tolerance = view_width * 0.02

        best: dict | None = None
        best_dist = float("inf")
        for ann in self._record.annotations:
            sample_index = ann.get("sample", 0)
            if 0 <= sample_index < len(self._record.time_axis):
                ann_time = float(self._record.time_axis[sample_index])
                dist = abs(ann_time - x_time)
                if dist < tolerance and dist < best_dist:
                    best_dist = dist
                    best = ann
        return best

    def _add_annotation(self, x_time: float, symbol: str) -> None:
        """Insert a new annotation at the local QRS peak nearest to *x_time*.

        The click position is first snapped to the highest signal point within
        a ±50 ms window so the label always lands on the actual QRS peak even
        when the user clicks somewhere in the middle of the complex.
        """
        if self._record is None:
            return
        raw_index = int(
            np.clip(
                np.searchsorted(self._record.time_axis, x_time),
                0,
                self._record.n_samples - 1,
            )
        )
        sample_index = self._snap_to_local_peak(raw_index)
        new_ann = {"sample": sample_index, "symbol": symbol}
        annotations = list(self._record.annotations) if self._record.annotations else []
        # Keep annotations sorted by sample index
        annotations.append(new_ann)
        annotations.sort(key=lambda a: a.get("sample", 0))
        self._record.annotations = annotations
        self._render()
        self.annotations_changed.emit()

    def _remove_annotation(self, ann: dict) -> None:
        """Remove *ann* from the record's annotation list."""
        if self._record is None or not self._record.annotations:
            return
        annotations = [a for a in self._record.annotations if a is not ann]
        self._record.annotations = annotations
        self._render()
        self.annotations_changed.emit()

    # ------------------------------------------------------------------ #
    #  Mouse event handling                                                #
    # ------------------------------------------------------------------ #

    def _on_mouse_clicked(self, event: tuple[object]) -> None:
        if self._record is None:
            return

        mouse_event = event[0]

        # --- Right-click handling ---
        if mouse_event.button() == Qt.MouseButton.RightButton:
            pos = mouse_event.scenePos()
            if not self.main_plot.sceneBoundingRect().contains(pos):
                return

            # Priority 1: click inside existing selection → existing menu
            if self._clicked_inside_selection(pos):
                self.selection_context_menu_requested.emit()
                mouse_event.accept()
                return

            # Priority 2: annotation context menu
            mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
            x_time = float(mouse_point.x())

            time_axis = self._record.time_axis
            if x_time < float(time_axis[0]) or x_time > float(time_axis[-1]):
                mouse_event.accept()
                return

            ann = self._annotation_near_time(x_time)
            global_pos = mouse_event.screenPos().toPoint()

            if ann is not None:
                self._show_annotation_remove_menu(ann, global_pos)
            else:
                self._show_annotation_add_menu(x_time, global_pos)

            mouse_event.accept()
            return

        # --- Left-click handling (unchanged) ---
        if mouse_event.button() != Qt.MouseButton.LeftButton:
            return

        pos = mouse_event.scenePos()
        if not self.main_plot.sceneBoundingRect().contains(pos):
            return

        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
        x_value = float(mouse_point.x())
        time_axis = self._record.time_axis
        if x_value < float(time_axis[0]) or x_value > float(time_axis[-1]):
            return

        sample_width = 1.0 / float(self._record.sampling_rate)
        view_start, view_end = self.main_plot.viewRange()[0]
        view_width = max(float(view_end - view_start), sample_width)
        total_width = max(float(time_axis[-1] - time_axis[0]), sample_width)
        selection_width = min(max(view_width * 0.15, sample_width * 8.0), total_width)
        self._set_selection_range(
            x_value - (selection_width / 2.0), x_value + (selection_width / 2.0)
        )
        mouse_event.accept()

    def _show_annotation_remove_menu(self, ann: dict, global_pos) -> None:
        """Show context menu offering to remove an existing annotation."""
        symbol = ann.get("symbol", "?")
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #FFFFFF; border: 1px solid #CFD8DC; border-radius: 6px; padding: 4px; }"
            "QMenu::item { padding: 6px 20px; font-size: 13px; color: #263238; border-radius: 4px; }"
            "QMenu::item:selected { background-color: #ECEFF1; }"
            "QMenu::separator { height: 1px; background: #CFD8DC; margin: 4px 8px; }"
        )

        title_action = menu.addAction(f"Adnotacja: '{symbol}'")
        title_action.setEnabled(False)
        menu.addSeparator()
        remove_action = menu.addAction("🗑  Usuń adnotację")

        chosen = menu.exec(global_pos)
        if chosen == remove_action:
            self._remove_annotation(ann)

    def _show_annotation_add_menu(self, x_time: float, global_pos) -> None:
        """Show context menu offering to add a new QRS annotation."""
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #FFFFFF; border: 1px solid #CFD8DC; border-radius: 6px; padding: 4px; }"
            "QMenu::item { padding: 6px 20px; font-size: 13px; color: #263238; border-radius: 4px; }"
            "QMenu::item:selected { background-color: #ECEFF1; }"
            "QMenu::separator { height: 1px; background: #CFD8DC; margin: 4px 8px; }"
        )

        title_action = menu.addAction("Dodaj adnotację QRS")
        title_action.setEnabled(False)
        menu.addSeparator()

        symbols = [
            ("N", "Rytm zatokowy normalny"),
            ("A", "Pobudzenie nadkomorowe (SVE)"),
            ("V", "Pobudzenie komorowe (VE)"),
            ("Q", "Nie do sklasyfikowania"),
        ]
        actions = {}
        for sym, desc in symbols:
            action = menu.addAction(f"Dodaj '{sym}'  —  {desc}")
            actions[action] = sym

        chosen = menu.exec(global_pos)
        if chosen in actions:
            self._add_annotation(x_time, actions[chosen])

    def _clicked_inside_selection(self, scene_pos) -> bool:
        selected_range = self.selected_time_range()
        if self._record is None or selected_range is None:
            return False
        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(scene_pos)
        click_time = float(mouse_point.x())
        start_time, end_time = selected_range
        return start_time <= click_time <= end_time

    def _on_plot_drag_selection(self, mouse_event) -> None:
        if self._record is None:
            mouse_event.ignore()
            return

        start_point = self.main_plot.plotItem.vb.mapToView(mouse_event.buttonDownPos())
        end_point = self.main_plot.plotItem.vb.mapToView(mouse_event.pos())
        self._set_selection_range(float(start_point.x()), float(end_point.x()))
        mouse_event.accept()

    def _set_selection_range(self, start_time: float, end_time: float) -> None:
        if self._record is None:
            return

        first_time = float(self._record.time_axis[0])
        last_time = float(self._record.time_axis[-1])
        sample_width = 1.0 / float(self._record.sampling_rate)

        bounded_start = max(first_time, min(start_time, end_time))
        bounded_end = min(last_time, max(start_time, end_time))
        if bounded_end - bounded_start < sample_width:
            midpoint = min(max((start_time + end_time) / 2.0, first_time), last_time)
            bounded_start = max(first_time, midpoint - (sample_width / 2.0))
            bounded_end = min(last_time, bounded_start + sample_width)
            bounded_start = max(first_time, bounded_end - sample_width)
        if bounded_end <= bounded_start:
            return

        self.selection_region.blockSignals(True)
        self.selection_region.setRegion((bounded_start, bounded_end))
        self.selection_region.blockSignals(False)
        self.selection_region.show()
        self._emit_selection_stats()

    def _emit_selection_stats(self) -> None:
        if self._record is None or not self.selection_region.isVisible():
            self.selection_changed.emit(None)
            return

        start_time, end_time = self.selection_region.getRegion()
        start_idx, end_idx = self._time_range_to_indices(start_time, end_time)
        if end_idx - start_idx <= 0:
            self.selection_changed.emit(None)
            return

        lead_idx = self._active_lead if self._view_mode == "single" else 0
        signal_segment = self._display_signal()[start_idx:end_idx, lead_idx]
        time_axis = self._record.time_axis[start_idx:end_idx]
        self.selection_changed.emit(compute_selection_stats(time_axis, signal_segment))

    def clear_selection(self) -> None:
        if not self.selection_region.isVisible():
            return
        self.selection_region.hide()
        self.selection_changed.emit(None)

    def selected_sample_range(self) -> tuple[int, int] | None:
        if self._record is None or not self.selection_region.isVisible():
            return None
        start_time, end_time = self.selection_region.getRegion()
        start_idx, end_idx = self._time_range_to_indices(start_time, end_time)
        if end_idx - start_idx <= 0:
            return None
        return start_idx, end_idx

    def selected_time_range(self) -> tuple[float, float] | None:
        if self._record is None or not self.selection_region.isVisible():
            return None
        start_time, end_time = self.selection_region.getRegion()
        clamped_start = max(float(self._record.time_axis[0]), float(start_time))
        clamped_end = min(float(self._record.time_axis[-1]), float(end_time))
        if clamped_end <= clamped_start:
            return None
        return clamped_start, clamped_end

    def reset_view(self) -> None:
        self._render()

    def go_to_start(self) -> None:
        pass

    def go_to_end(self) -> None:
        pass

    def current_window_seconds(self) -> int | None:
        return self._window_seconds

    def visible_time_range(self) -> tuple[float, float] | None:
        if self._record is None:
            return None
        start_time, end_time = self.main_plot.viewRange()[0]
        bounded_start = max(float(self._record.time_axis[0]), float(start_time))
        bounded_end = min(float(self._record.time_axis[-1]), float(end_time))
        if bounded_end <= bounded_start:
            return None
        return bounded_start, bounded_end
