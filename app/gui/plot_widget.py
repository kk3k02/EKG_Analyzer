from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

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


class ECGPlotWidget(QWidget):
    """Interactive Stage 1 ECG viewer built around pyqtgraph.
    Selection statistics emitted from this widget refer only to the active lead
    in single-lead mode or to the lead nearest the interaction point in stacked
    mode. The widget does not provide clinical annotations or multi-lead
    diagnostic measurements yet..
    """

    cursor_changed = Signal(object)
    selection_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 20, 0, 0)
        layout.setSpacing(10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        pg.setConfigOptions(antialias=False)

        self.main_plot = pg.PlotWidget()
        self.main_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_plot.setBackground("w")
        self.main_plot.showGrid(x=True, y=True, alpha=0.25)
        self.main_plot.setLabel("bottom", "Czas", units="s")
        self.main_plot.setLabel("left", "Amplituda")
        self.main_plot.addLegend(offset=(10, 10))

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

        layout.addWidget(self.main_plot, stretch=2)
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

        self._current_playback_time = 0.0
        self._cursor_pos = 0.0

        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#D32F2F", width=4))
        self.main_plot.addItem(self.cursor_line, ignoreBounds=True)

        self.selection_region = pg.LinearRegionItem(values=(0, 1), movable=True, brush=(200, 30, 30, 40))
        self.selection_region.setZValue(10)
        self.selection_region.sigRegionChanged.connect(self._emit_selection_stats)
        self.main_plot.addItem(self.selection_region, ignoreBounds=True)
        self.selection_region.hide()

        self._mouse_proxy = pg.SignalProxy(self.main_plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)
        self._main_plot_mouse_proxy = pg.SignalProxy(
            self.main_plot.scene().sigMouseClicked,
            rateLimit=30,
            slot=self._on_mouse_clicked,
        )

        self._overview_update_timer = QTimer(self)
        self._overview_update_timer.setSingleShot(True)
        self._overview_update_timer.setInterval(120)
        self._overview_update_timer.timeout.connect(self.update_frequency_overview_plot)

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
        self._lead_visibility = {index: True for index in range(record.n_leads)} if record else {}
        self._cursor_pos = float(record.time_axis[0]) if record else 0.0
        self._last_frequency_cache_key = None
        self._render()

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

        self._current_playback_time = start_time
        self._cursor_pos = start_time

        win_new = window_seconds if window_seconds > 0 else 5.0
        overlap = 1.0

        page_idx = int(start_time / win_new)
        page_start_raw = page_idx * win_new

        view_start = max(float(self._record.time_axis[0]), page_start_raw - overlap)
        view_end = view_start + win_new + (overlap if page_idx > 0 else 0)

        self.main_plot.setXRange(view_start, view_end, padding=0.0)
        self.cursor_line.setPos(start_time)

        self._update_revealed_data(view_start, start_time)
        self._schedule_frequency_overview_update()

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
            y_data = display_signal[start_idx:end_idx, lead_idx] + offsets.get(lead_idx, 0.0)
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
            self._show_frequency_overview_message("Wczytaj plik, aby zobaczyc analize.")
            return

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
                label = pg.TextItem(text=self._record.lead_names[lead_idx], color="#D32F2F", anchor=(0, 0.5))
                label.setPos(float(self._record.time_axis[0]), offsets.get(lead_idx, 0.0))
                self.main_plot.addItem(label)

        if self._record.annotations:
            lead_idx = self._active_lead if self._view_mode == "single" else 0
            offset = offsets.get(lead_idx, 0.0)
            for ann in self._record.annotations:
                sample_index = ann.get("sample", 0)
                if 0 <= sample_index < len(self._record.time_axis):
                    txt = pg.TextItem(text=ann.get("symbol", "?"), color="#000000", anchor=(0.5, 1))
                    txt.setPos(
                        float(self._record.time_axis[sample_index]),
                        float(display_signal[sample_index, lead_idx]) + offset + 0.1,
                    )
                    txt.hide()
                    self.main_plot.addItem(txt)
                    self._annotation_items.append(txt)

        self._schedule_frequency_overview_update(immediate=True)

    def update_frequency_overview_plot(self) -> None:
        if self._record is None:
            return

        start, end = self.main_plot.viewRange()[0]
        start_idx, end_idx = self._time_range_to_indices(start, end)
        lead_idx = self._get_frequency_overview_lead_index()
        if lead_idx is None:
            return

        cache_key = (start_idx, end_idx, lead_idx, self.log_scale_checkbox.isChecked(), self._max_frequency_hz)
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

    def _render_frequency_overview(self, result: FrequencyOverviewResult, lead_name: str) -> None:
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

    def _compute_offsets(self, signal: np.ndarray, visible_indices: list[int]) -> dict[int, float]:
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

    def _display_signal(self) -> np.ndarray:
        if self._filtering_active and self._preview_signal is not None:
            return self._preview_signal
        return self._record.signal

    def _visible_indices(self) -> list[int]:
        if self._record is None:
            return []
        if self._view_mode == "single":
            return [self._active_lead]
        return [i for i in range(self._record.n_leads) if self._lead_visibility.get(i, True)]

    def _time_range_to_indices(self, start_time: float, end_time: float) -> tuple[int, int]:
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
        idx = int(np.clip(np.searchsorted(self._record.time_axis, x_value), 0, self._record.n_samples - 1))
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

    def _on_mouse_clicked(self, event: tuple[object]) -> None:
        pass

    def _emit_selection_stats(self) -> None:
        if self._record is None or not self.selection_region.isVisible():
            return

        start_time, end_time = self.selection_region.getRegion()
        start_idx, end_idx = self._time_range_to_indices(start_time, end_time)
        if end_idx - start_idx <= 0:
            return

        lead_idx = self._active_lead if self._view_mode == "single" else 0
        signal_segment = self._display_signal()[start_idx:end_idx, lead_idx]
        time_axis = self._record.time_axis[start_idx:end_idx]
        self.selection_changed.emit(compute_selection_stats(time_axis, signal_segment))

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
