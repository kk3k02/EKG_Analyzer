from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal, QTimer
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from app.models.ecg_record import ECGRecord
from app.services.frequency_analysis import FrequencyAnalysisService


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
    diagnostic measurements yet.
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
        self.main_plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.main_plot.setBackground("w")
        self.main_plot.showGrid(x=True, y=True, alpha=0.25)
        self.main_plot.setLabel("bottom", "Czas", units="s")
        self.main_plot.setLabel("left", "Amplituda (Monitor)")

        self.frequency_plot = pg.PlotWidget()
        self.frequency_plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.frequency_plot.setBackground("w")
        self.frequency_plot.showGrid(x=True, y=True, alpha=0.25)
        self.frequency_plot.setLabel("bottom", "Częstotliwość", units="Hz")
        self.frequency_plot.setLabel("left", "Amplituda")
        self.frequency_plot.setXRange(0, 100)

        layout.addWidget(self.main_plot, stretch=2)
        layout.addWidget(self.frequency_plot, stretch=1)

        self._record: ECGRecord | None = None
        self._preview_signal: np.ndarray | None = None
        self._raw_visible = True
        self._filtered_visible = False
        self._view_mode = "stacked"
        self._active_lead = 0
        self._lead_visibility: dict[int, bool] = {}

        self._monitor_curves: list[pg.PlotDataItem] = []

        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(30)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        self._playback_speed = 1.0
        self._window_size = 5.0
        self._current_page_start = 0.0
        self._cursor_pos = 0.0

        self.cursor_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#D32F2F", width=3)
        )
        self.main_plot.addItem(self.cursor_line, ignoreBounds=True)

        self._mouse_proxy = pg.SignalProxy(
            self.main_plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )

    def set_record(
        self, record: ECGRecord | None, preview_signal: np.ndarray | None = None
    ) -> None:
        self._record = record
        self._preview_signal = preview_signal
        if record:
            self._current_page_start = float(record.time_axis[0])
            self._cursor_pos = self._current_page_start
            self._lead_visibility = {index: True for index in range(record.n_leads)}
        self._render()

    def set_playback(self, active: bool) -> None:
        if active and self._record:
            self._playback_timer.start()
        else:
            self._playback_timer.stop()

    def set_playback_speed(self, speed: float) -> None:
        self._playback_speed = speed

    def _on_playback_tick(self) -> None:
        if self._record is None:
            return

        dt = (self._playback_timer.interval() / 5000.0) * self._playback_speed
        self._cursor_pos += dt

        page_end = self._current_page_start + self._window_size
        if self._cursor_pos >= page_end:
            self._current_page_start = page_end
            self._cursor_pos = self._current_page_start

            if self._current_page_start >= self._record.time_axis[-1]:
                self._current_page_start = float(self._record.time_axis[0])
                self._cursor_pos = self._current_page_start

            self._update_window_range()
            self._update_fft_for_current_window()

        self.cursor_line.setPos(self._cursor_pos)

        self._update_revealed_data()
        self._emit_cursor_info_at_pos(self._cursor_pos)

    def _update_revealed_data(self) -> None:
        if self._record is None:
            return

        time_axis = self._record.time_axis
        mask = (time_axis >= self._current_page_start) & (time_axis <= self._cursor_pos)

        x_data = time_axis[mask]
        preview_signal = (
            self._preview_signal
            if self._preview_signal is not None
            else self._record.signal
        )

        visible_indices = [
            i for i in range(self._record.n_leads) if self._lead_visibility.get(i, True)
        ]
        if self._view_mode == "single":
            visible_indices = [self._active_lead]

        offsets = self._compute_offsets(preview_signal, visible_indices)

        for i, lead_idx in enumerate(visible_indices):
            if i < len(self._monitor_curves):
                y_data = preview_signal[mask, lead_idx] + offsets.get(lead_idx, 0.0)
                self._monitor_curves[i].setData(x_data, y_data)

    def _update_window_range(self) -> None:
        if self._record:
            start = self._current_page_start
            end = start + self._window_size
            self.main_plot.setXRange(start, end, padding=0)

    def _update_fft_for_current_window(self) -> None:
        if self._record is None:
            return
        start = self._current_page_start
        end = start + self._window_size
        mask = (self._record.time_axis >= start) & (self._record.time_axis <= end)

        if not np.any(mask):
            return

        lead_idx = self._active_lead if self._view_mode == "single" else 0
        preview_signal = (
            self._preview_signal
            if self._preview_signal is not None
            else self._record.signal
        )
        sig_chunk = preview_signal[mask, lead_idx]

        if len(sig_chunk) > 10:
            freqs, amps = FrequencyAnalysisService.compute_fft(
                sig_chunk, self._record.sampling_rate
            )
            self.frequency_plot.clear()
            self.frequency_plot.plot(
                freqs, amps, pen=pg.mkPen(color="#8B0000", width=2.0)
            )
            self.frequency_plot.setTitle(
                f"Widmo (FFT) dla okna {start:.1f}s - {end:.1f}s"
            )

    def _render(self) -> None:
        self.main_plot.clear()
        self._monitor_curves.clear()
        self.main_plot.addItem(self.cursor_line, ignoreBounds=True)

        if self._record is None:
            return

        preview_signal = (
            self._preview_signal
            if self._preview_signal is not None
            else self._record.signal
        )
        visible_indices = [
            i for i in range(self._record.n_leads) if self._lead_visibility.get(i, True)
        ]
        if self._view_mode == "single":
            visible_indices = [self._active_lead]

        offsets = self._compute_offsets(preview_signal, visible_indices)

        for lead_idx in visible_indices:
            curve = pg.PlotDataItem(
                pen=pg.mkPen(color="#D32F2F", width=2.5)
            )  # Intensywny czerwony
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

        self._update_window_range()
        self._update_fft_for_current_window()
        self._update_revealed_data()

    def _compute_offsets(
        self, signal: np.ndarray, visible_indices: list[int]
    ) -> dict[int, float]:
        if self._view_mode == "single":
            return {self._active_lead: 0.0}
        spacing = 3.0
        return {idx: -i * spacing for i, idx in enumerate(visible_indices)}

    def _on_mouse_moved(self, event: tuple[object]) -> None:
        if self._record is None:
            return
        pos = event[0]
        if not self.main_plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
        x_val = float(mouse_point.x())

        self._emit_cursor_info_at_pos(x_val)

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

    def set_raw_visible(self, v: bool) -> None:
        self._raw_visible = v
        self._render()

    def set_filtered_visible(self, v: bool) -> None:
        self._filtered_visible = v
        self._render()

    def set_grid_visible(self, v: bool) -> None:
        self.main_plot.showGrid(x=v, y=v)

    def set_view_mode(self, m: str) -> None:
        self._view_mode = m
        self._render()

    def set_active_lead(self, i: int) -> None:
        self._active_lead = i
        self._render()

    def set_lead_visibility(self, v: dict[int, bool]) -> None:
        self._lead_visibility = v
        self._render()

    def reset_view(self) -> None:
        self._current_page_start = self._record.time_axis[0] if self._record else 0.0
        self._render()

    def go_to_start(self) -> None:
        self._current_page_start = self._record.time_axis[0] if self._record else 0.0
        self._render()

    def go_to_end(self) -> None:
        self._current_page_start = (
            max(0, self._record.time_axis[-1] - self._window_size)
            if self._record
            else 0.0
        )
        self._render()

    def set_window_seconds(self, s: int | None) -> None:
        if s:
            self._window_size = float(s)
        self._render()
