from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
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

        pg.setConfigOptions(antialias=True)

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
        self.overview_title = QLabel("Przegląd częstotliwości (FFT)", self)
        self.overview_title.setStyleSheet("font-weight: 600;")
        self.log_scale_checkbox = QCheckBox("Skala log", self)
        self.log_scale_checkbox.toggled.connect(self.update_frequency_overview_plot)
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
        self.overview_plot.setLabel("bottom", "Częstotliwość", units="Hz")
        self.overview_plot.setLabel("left", "Amplituda")

        self.overview_empty_label = QLabel("Wczytaj plik, aby zobaczyć analizę.", self)
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
        if self._record is None: return
        
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
        
        self.update_frequency_overview_plot()

    def _update_revealed_data(self, view_start: float, cursor_pos: float) -> None:
        if self._record is None: return
        time_axis = self._record.time_axis

        mask = (time_axis >= view_start) & (time_axis <= cursor_pos)
        
        x_data = time_axis[mask]
        display_signal = self._preview_signal if self._filtering_active and self._preview_signal is not None else self._record.signal
        
        visible_indices = [i for i in range(self._record.n_leads) if self._lead_visibility.get(i, True)]
        if self._view_mode == "single": visible_indices = [self._active_lead]
        offsets = self._compute_offsets(display_signal, visible_indices)

        for i, lead_idx in enumerate(visible_indices):
            if i < len(self._monitor_curves):
                y_data = display_signal[mask, lead_idx] + offsets.get(lead_idx, 0.0)
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
            self._show_frequency_overview_message("Wczytaj plik, aby zobaczyć analizę.")
            return

        display_signal = self._preview_signal if self._filtering_active and self._preview_signal is not None else self._record.signal
        visible_indices = [i for i in range(self._record.n_leads) if self._lead_visibility.get(i, True)]
        if self._view_mode == "single": visible_indices = [self._active_lead]
        offsets = self._compute_offsets(display_signal, visible_indices)

        for i, lead_idx in enumerate(visible_indices):
            curve = pg.PlotDataItem(pen=pg.mkPen(color="#D32F2F", width=3.5))
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
                s = ann.get("sample", 0)
                if 0 <= s < len(self._record.time_axis):
                    txt = pg.TextItem(text=ann.get("symbol", "?"), color="#000000", anchor=(0.5, 1))
                    txt.setPos(float(self._record.time_axis[s]), float(display_signal[s, lead_idx]) + offset + 0.1)
                    txt.hide()
                    self.main_plot.addItem(txt)
                    self._annotation_items.append(txt)

        self.update_frequency_overview_plot()

    def update_frequency_overview_plot(self) -> None:
        if self._record is None: return
        
        start, end = self.main_plot.viewRange()[0]
        mask = (self._record.time_axis >= start) & (self._record.time_axis <= end)
        
        lead_idx = self._get_frequency_overview_lead_index()
        if lead_idx is None: return

        display_signal = self._preview_signal if self._filtering_active and self._preview_signal is not None else self._record.signal
        signal_segment = display_signal[mask, lead_idx]
        
        if signal_segment.size < 16:
            self._show_frequency_overview_message("Zbyt mało danych do FFT.")
            return

        result = compute_frequency_overview(signal_segment, self._record.sampling_rate, max_frequency_hz=self._max_frequency_hz)
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

        self.overview_plot.plot(result.frequencies_hz, values, pen=pg.mkPen(color="#00429d", width=3.0))
        self.overview_plot.setXRange(0.0, self._max_frequency_hz, padding=0.01)
        self.overview_stack.setCurrentWidget(self.overview_plot)

    def _compute_offsets(self, signal: np.ndarray, visible_indices: list[int]) -> dict[int, float]:
        if self._view_mode == "single": return {self._active_lead: 0.0}
        return {idx: -i * 3.0 for i, idx in enumerate(visible_indices)}

    def _get_frequency_overview_lead_index(self) -> int | None:
        if self._record is None: return None
        if self._view_mode == "single": return self._active_lead
        vis = [i for i in range(self._record.n_leads) if self._lead_visibility.get(i, True)]
        return vis[0] if vis else None

    def _show_frequency_overview_message(self, message: str) -> None:
        self.overview_empty_label.setText(message)
        self.overview_stack.setCurrentWidget(self.overview_empty_label)

    def _on_mouse_moved(self, event: tuple[object]) -> None:
        if self._record is None: return
        pos = event[0]
        if not self.main_plot.sceneBoundingRect().contains(pos): return
        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
        self._emit_cursor_info_at_pos(float(mouse_point.x()))

    def _emit_cursor_info_at_pos(self, x_value: float) -> None:
        if self._record is None: return
        idx = int(np.clip(np.searchsorted(self._record.time_axis, x_value), 0, self._record.n_samples - 1))
        lead_idx = self._active_lead if self._view_mode == "single" else 0
        if x_value < self._record.time_axis[0] or x_value > self._record.time_axis[-1]: return
        self.cursor_changed.emit(CursorInfo(
            time_value=float(self._record.time_axis[idx]),
            sample_index=idx,
            amplitude=float(self._record.signal[idx, lead_idx]),
            lead_name=self._record.lead_names[lead_idx]
        ))

    def _on_mouse_clicked(self, event: tuple[object]) -> None: pass
    def _emit_selection_stats(self) -> None: pass
    def reset_view(self) -> None: self._render()
    def go_to_start(self) -> None: pass
    def go_to_end(self) -> None: pass
    def current_window_seconds(self) -> int | None: return self._window_seconds
