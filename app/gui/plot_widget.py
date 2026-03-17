from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSizePolicy, QStackedLayout, QVBoxLayout, QWidget

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
    diagnostic measurements yet.
    """

    cursor_changed = Signal(object)
    selection_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
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
        self.overview_title = QLabel("Przeglad czestotliwosci", self)
        self.overview_title.setStyleSheet("font-weight: 600;")
        self.log_scale_checkbox = QCheckBox("Skala log", self)
        self.log_scale_checkbox.toggled.connect(self.update_frequency_overview_plot)
        overview_header.addWidget(self.overview_title)
        overview_header.addStretch(1)
        overview_header.addWidget(self.log_scale_checkbox)
        overview_layout.addLayout(overview_header)

        self.overview_plot = pg.PlotWidget()
        self.overview_plot.setMinimumHeight(90)
        self.overview_plot.setMaximumHeight(140)
        self.overview_plot.setBackground("w")
        self.overview_plot.setMouseEnabled(x=False, y=False)
        self.overview_plot.showGrid(x=True, y=True, alpha=0.15)
        self.overview_plot.setLabel("bottom", "Czestotliwosc", units="Hz")
        self.overview_plot.setLabel("left", "Gestosc mocy widmowej")

        self.overview_empty_label = QLabel("Wczytaj plik, aby zobaczyc przeglad czestotliwosci.", self)
        self.overview_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overview_empty_label.setStyleSheet("color: #666666;")

        self.overview_stack = QStackedLayout()
        self.overview_stack.addWidget(self.overview_plot)
        self.overview_stack.addWidget(self.overview_empty_label)
        overview_layout.addLayout(self.overview_stack)

        layout.addWidget(self.main_plot, stretch=1)
        layout.addWidget(overview_container)

        self._record: ECGRecord | None = None
        self._preview_signal: np.ndarray | None = None
        self._filtering_active = False
        self._raw_visible = False
        self._filtered_visible = True
        self._show_grid = True
        self._view_mode = "stacked"
        self._active_lead = 0
        self._lead_visibility: dict[int, bool] = {}
        self._curves: list[pg.PlotDataItem] = []
        self._overview_plot_item: pg.PlotDataItem | None = None
        self._frequency_markers: list[pg.InfiniteLine] = []
        self._overview_mode = "frequency"
        self._max_frequency_hz = DEFAULT_MAX_FREQUENCY_HZ

        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#8B0000", width=1))
        self.main_plot.addItem(self.cursor_line, ignoreBounds=True)

        self.selection_region = pg.LinearRegionItem(values=(0, 1), movable=True, brush=(200, 30, 30, 40))
        self.selection_region.setZValue(10)
        self.selection_region.sigRegionChanged.connect(self._emit_selection_stats)
        self.main_plot.addItem(self.selection_region, ignoreBounds=True)
        self.selection_region.hide()

        self.main_plot.sigXRangeChanged.connect(self._handle_main_plot_range_changed)
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
        if self._record is None:
            return
        if seconds is None:
            self.main_plot.setXRange(float(self._record.time_axis[0]), float(self._record.time_axis[-1]), padding=0.01)
            return
        start = float(self._record.time_axis[0])
        end = min(start + float(seconds), float(self._record.time_axis[-1]))
        self.main_plot.setXRange(start, end, padding=0.0)

    def reset_view(self) -> None:
        if self._record is None:
            return
        self.main_plot.enableAutoRange()
        self.main_plot.setXRange(float(self._record.time_axis[0]), float(self._record.time_axis[-1]), padding=0.01)

    def go_to_start(self) -> None:
        self.set_window_seconds(10)

    def go_to_end(self) -> None:
        if self._record is None:
            return
        current_range = self.main_plot.viewRange()[0]
        width = current_range[1] - current_range[0]
        end = float(self._record.time_axis[-1])
        start = max(float(self._record.time_axis[0]), end - width)
        self.main_plot.setXRange(start, end, padding=0.0)

    def update_preview_signal(self, preview_signal: np.ndarray | None) -> None:
        self._preview_signal = preview_signal
        self._render()

    def _render(self) -> None:
        self.main_plot.clear()
        self.overview_plot.clear()
        self.main_plot.addItem(self.cursor_line, ignoreBounds=True)
        self.main_plot.addItem(self.selection_region, ignoreBounds=True)
        self._curves.clear()
        self._overview_plot_item = None
        self._frequency_markers.clear()

        if self._record is None:
            self.selection_region.hide()
            self._show_frequency_overview_message("Wczytaj plik, aby zobaczyc przeglad czestotliwosci.")
            return

        self.main_plot.showGrid(x=self._show_grid, y=self._show_grid, alpha=0.25)
        time_axis = self._record.time_axis
        raw_signal = self._record.signal
        preview_signal = self._preview_signal if self._preview_signal is not None else raw_signal
        primary_signal = preview_signal if self._filtering_active else raw_signal
        colors = ["#00429d", "#73a2c6", "#eeb479", "#93003a", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

        visible_indices = [index for index in range(self._record.n_leads) if self._lead_visibility.get(index, True)]
        if self._view_mode == "single":
            visible_indices = [self._active_lead]

        stacked_offsets = self._compute_offsets(primary_signal, visible_indices)

        for order, lead_index in enumerate(visible_indices):
            color = colors[order % len(colors)]
            lead_name = self._record.lead_names[lead_index]
            offset = stacked_offsets.get(lead_index, 0.0)
            if self._filtered_visible or not self._filtering_active:
                self.main_plot.plot(
                    time_axis,
                    primary_signal[:, lead_index] + offset,
                    pen=pg.mkPen(color=color, width=1.2),
                    name=f"{lead_name} przetworzony" if self._filtering_active else f"{lead_name} surowy",
                )
            if self._filtering_active and self._raw_visible:
                self.main_plot.plot(
                    time_axis,
                    raw_signal[:, lead_index] + offset,
                    pen=pg.mkPen(color="#444444", width=1.0, style=Qt.PenStyle.DashLine),
                    name=f"{lead_name} surowy - nakladka",
                )

            if self._view_mode == "stacked":
                label = pg.TextItem(text=lead_name, color=color, anchor=(0, 0.5))
                label.setPos(float(time_axis[0]), offset)
                self.main_plot.addItem(label)

        self.selection_region.setRegion((float(time_axis[0]), min(float(time_axis[-1]), float(time_axis[0]) + 1.0)))
        self.selection_region.show()
        self.main_plot.setXRange(float(time_axis[0]), min(float(time_axis[-1]), float(time_axis[0]) + 10.0), padding=0.01)
        self.update_frequency_overview_plot()
        self._emit_selection_stats()

    def _compute_offsets(self, signal: np.ndarray, visible_indices: list[int]) -> dict[int, float]:
        if self._view_mode == "single":
            return {self._active_lead: 0.0}
        if not visible_indices:
            return {}
        amplitudes = [np.nanmax(signal[:, index]) - np.nanmin(signal[:, index]) for index in visible_indices]
        base_spacing = float(max(amplitudes) if amplitudes else 1.0)
        spacing = max(base_spacing * 1.5, 1.0)
        return {lead_index: -order * spacing for order, lead_index in enumerate(visible_indices)}

    def _on_mouse_moved(self, event: tuple[object]) -> None:
        if self._record is None:
            return
        pos = event[0]
        if not self.main_plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
        x_value = float(mouse_point.x())
        sample_index = int(np.clip(np.searchsorted(self._record.time_axis, x_value), 0, self._record.n_samples - 1))
        lead_index = self._active_lead if self._view_mode == "single" else self._nearest_visible_lead(mouse_point.y())
        display_signal = self._preview_signal if self._filtering_active and self._preview_signal is not None else self._record.signal
        amplitude = float(display_signal[sample_index, lead_index])
        self.cursor_line.setPos(self._record.time_axis[sample_index])
        self.cursor_changed.emit(
            CursorInfo(
                time_value=float(self._record.time_axis[sample_index]),
                sample_index=sample_index,
                amplitude=amplitude,
                lead_name=self._record.lead_names[lead_index],
            )
        )

    def _nearest_visible_lead(self, y_value: float) -> int:
        if self._record is None:
            return 0
        if self._view_mode == "single":
            return self._active_lead
        visible_indices = [index for index in range(self._record.n_leads) if self._lead_visibility.get(index, True)]
        analysis_signal = self._preview_signal if self._filtering_active and self._preview_signal is not None else self._record.signal
        offsets = self._compute_offsets(analysis_signal, visible_indices)
        if not offsets:
            return 0
        return min(offsets, key=lambda index: abs(offsets[index] - y_value))

    def _on_mouse_clicked(self, event: tuple[object]) -> None:
        if self._record is None:
            return
        mouse_event = event[0]
        if mouse_event.button() != Qt.MouseButton.LeftButton or not mouse_event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return
        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(mouse_event.scenePos())
        center = float(np.clip(mouse_point.x(), self._record.time_axis[0], self._record.time_axis[-1]))
        half_width = min(1.0, self._record.duration_seconds / 4.0 if self._record.duration_seconds else 0.5)
        start = max(float(self._record.time_axis[0]), center - half_width)
        end = min(float(self._record.time_axis[-1]), center + half_width)
        self.selection_region.setRegion((start, end))
        self.selection_region.show()
        self._emit_selection_stats()

    def _emit_selection_stats(self) -> None:
        if self._record is None or not self.selection_region.isVisible():
            return
        start, end = self.selection_region.getRegion()
        mask = (self._record.time_axis >= start) & (self._record.time_axis <= end)
        if not np.any(mask):
            return
        # Stage 1 keeps selection stats intentionally simple: one lead only.
        lead_index = self._active_lead if self._view_mode == "single" else self._nearest_visible_lead(0.0)
        analysis_signal = self._preview_signal if self._filtering_active and self._preview_signal is not None else self._record.signal
        stats = compute_selection_stats(self._record.time_axis[mask], analysis_signal[mask, lead_index])
        self.selection_changed.emit(stats)

    def get_visible_signal_segment(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._record is None:
            return None, None
        lead_index = self._get_frequency_overview_lead_index()
        if lead_index is None:
            return None, None
        start, end = self._get_visible_time_range()
        sample_slice = self._time_range_to_sample_slice(start, end)
        analysis_signal = self._preview_signal if self._filtering_active and self._preview_signal is not None else self._record.signal
        return self._record.time_axis[sample_slice], analysis_signal[sample_slice, lead_index]

    def update_frequency_overview_plot(self) -> None:
        if self._overview_mode != "frequency":
            self._show_frequency_overview_message("Tryb przegladu jest niedostepny.")
            return
        if self._record is None:
            self._show_frequency_overview_message("Wczytaj plik, aby zobaczyc przeglad czestotliwosci.")
            return

        lead_index = self._get_frequency_overview_lead_index()
        if lead_index is None:
            self._show_frequency_overview_message("Wybierz odprowadzenie, aby zobaczyc przeglad czestotliwosci.")
            return

        _, signal_segment = self.get_visible_signal_segment()
        if signal_segment is None or signal_segment.size == 0:
            self._show_frequency_overview_message("Widoczny fragment sygnalu jest pusty.")
            return

        result = compute_frequency_overview(
            signal_segment,
            self._record.sampling_rate,
            max_frequency_hz=self._max_frequency_hz,
        )
        if result.message is not None:
            self._show_frequency_overview_message(result.message)
            return

        self._render_frequency_overview(result, self._record.lead_names[lead_index])

    def _render_frequency_overview(self, result: FrequencyOverviewResult, lead_name: str) -> None:
        self.overview_plot.clear()
        self.overview_plot.showGrid(x=True, y=True, alpha=0.15)
        self.overview_plot.setLabel("bottom", "Czestotliwosc", units="Hz")
        self.overview_plot.setLabel("left", result.y_label)
        self.overview_plot.setTitle(f"{lead_name} ({result.method.upper()})")

        values = np.asarray(result.values, dtype=float)
        if self.log_scale_checkbox.isChecked():
            values = np.maximum(values, np.finfo(float).tiny)
            self.overview_plot.setLogMode(x=False, y=True)
        else:
            self.overview_plot.setLogMode(x=False, y=False)

        self._overview_plot_item = self.overview_plot.plot(
            result.frequencies_hz,
            values,
            pen=pg.mkPen(color="#00429d", width=1.5),
        )
        self.overview_plot.setXRange(0.0, self._max_frequency_hz, padding=0.01)
        self._add_frequency_markers()
        self.overview_stack.setCurrentWidget(self.overview_plot)

    def _add_frequency_markers(self) -> None:
        self._frequency_markers.clear()
        for frequency_hz, color in ((50.0, "#cc5500"), (60.0, "#666666")):
            if frequency_hz > self._max_frequency_hz:
                continue
            marker = pg.InfiniteLine(
                pos=frequency_hz,
                angle=90,
                movable=False,
                pen=pg.mkPen(color=color, width=1, style=Qt.PenStyle.DashLine),
            )
            self.overview_plot.addItem(marker, ignoreBounds=True)
            self._frequency_markers.append(marker)

    def _show_frequency_overview_message(self, message: str) -> None:
        self.overview_plot.clear()
        self.overview_plot.setTitle("")
        self.overview_empty_label.setText(message)
        self.overview_stack.setCurrentWidget(self.overview_empty_label)

    def _get_frequency_overview_lead_index(self) -> int | None:
        if self._record is None:
            return None
        if self._view_mode == "single":
            if 0 <= self._active_lead < self._record.n_leads:
                return self._active_lead
            return None
        visible_indices = [index for index in range(self._record.n_leads) if self._lead_visibility.get(index, True)]
        return visible_indices[0] if visible_indices else None

    def _get_visible_time_range(self) -> tuple[float, float]:
        if self._record is None:
            return 0.0, 0.0
        view_start, view_end = self.main_plot.viewRange()[0]
        record_start = float(self._record.time_axis[0])
        record_end = float(self._record.time_axis[-1])
        return max(record_start, float(view_start)), min(record_end, float(view_end))

    def _time_range_to_sample_slice(self, start_time: float, end_time: float) -> slice:
        if self._record is None:
            return slice(0, 0)
        time_axis = self._record.time_axis
        start_index = int(np.searchsorted(time_axis, start_time, side="left"))
        end_index = int(np.searchsorted(time_axis, end_time, side="right"))
        start_index = max(0, min(start_index, self._record.n_samples))
        end_index = max(start_index + 1, min(end_index, self._record.n_samples))
        return slice(start_index, end_index)

    def _handle_main_plot_range_changed(self, *_args: object) -> None:
        self.update_frequency_overview_plot()
