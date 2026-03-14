from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from app.models.ecg_record import ECGRecord
from app.services.selection_stats import SelectionStats, compute_selection_stats


@dataclass(slots=True)
class CursorInfo:
    time_value: float
    sample_index: int
    amplitude: float
    lead_name: str


class ECGPlotWidget(QWidget):
    cursor_changed = Signal(object)
    selection_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        pg.setConfigOptions(antialias=False)

        self.main_plot = pg.PlotWidget()
        self.main_plot.setBackground("w")
        self.main_plot.showGrid(x=True, y=True, alpha=0.25)
        self.main_plot.setLabel("bottom", "Time", units="s")
        self.main_plot.setLabel("left", "Amplitude")
        self.main_plot.addLegend(offset=(10, 10))

        self.overview_plot = pg.PlotWidget(maximumHeight=120)
        self.overview_plot.setBackground("w")
        self.overview_plot.setMouseEnabled(x=False, y=False)
        self.overview_plot.hideAxis("left")
        self.overview_plot.setLabel("bottom", "Overview", units="s")

        layout.addWidget(self.main_plot, stretch=1)
        layout.addWidget(self.overview_plot)

        self._record: ECGRecord | None = None
        self._preview_signal: np.ndarray | None = None
        self._raw_visible = True
        self._filtered_visible = False
        self._show_grid = True
        self._view_mode = "stacked"
        self._active_lead = 0
        self._lead_visibility: dict[int, bool] = {}
        self._curves: list[pg.PlotDataItem] = []
        self._overview_curves: list[pg.PlotDataItem] = []

        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#8B0000", width=1))
        self.main_plot.addItem(self.cursor_line, ignoreBounds=True)

        self.selection_region = pg.LinearRegionItem(values=(0, 1), movable=True, brush=(200, 30, 30, 40))
        self.selection_region.setZValue(10)
        self.selection_region.sigRegionChanged.connect(self._emit_selection_stats)
        self.main_plot.addItem(self.selection_region, ignoreBounds=True)
        self.selection_region.hide()

        self.overview_region = pg.LinearRegionItem(values=(0, 1), movable=True, brush=(30, 120, 200, 30))
        self.overview_region.sigRegionChanged.connect(self._sync_main_from_overview)
        self.overview_plot.addItem(self.overview_region)
        self.overview_region.hide()

        self.main_plot.sigXRangeChanged.connect(self._sync_overview_from_main)
        self._mouse_proxy = pg.SignalProxy(self.main_plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)
        self._main_plot_mouse_proxy = pg.SignalProxy(
            self.main_plot.scene().sigMouseClicked,
            rateLimit=30,
            slot=self._on_mouse_clicked,
        )

    def set_record(self, record: ECGRecord | None, preview_signal: np.ndarray | None = None) -> None:
        self._record = record
        self._preview_signal = preview_signal
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
        self.overview_plot.addItem(self.overview_region)
        self._curves.clear()
        self._overview_curves.clear()

        if self._record is None:
            self.selection_region.hide()
            self.overview_region.hide()
            return

        self.main_plot.showGrid(x=self._show_grid, y=self._show_grid, alpha=0.25)
        time_axis = self._record.time_axis
        raw_signal = self._record.signal
        preview_signal = self._preview_signal if self._preview_signal is not None else raw_signal
        colors = ["#00429d", "#73a2c6", "#eeb479", "#93003a", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

        visible_indices = [index for index in range(self._record.n_leads) if self._lead_visibility.get(index, True)]
        if self._view_mode == "single":
            visible_indices = [self._active_lead]

        stacked_offsets = self._compute_offsets(preview_signal, visible_indices)

        for order, lead_index in enumerate(visible_indices):
            color = colors[order % len(colors)]
            lead_name = self._record.lead_names[lead_index]
            offset = stacked_offsets.get(lead_index, 0.0)
            if self._raw_visible:
                self.main_plot.plot(
                    time_axis,
                    raw_signal[:, lead_index] + offset,
                    pen=pg.mkPen(color=color, width=1.2),
                    name=f"{lead_name} raw",
                )
            if self._filtered_visible:
                self.main_plot.plot(
                    time_axis,
                    preview_signal[:, lead_index] + offset,
                    pen=pg.mkPen(color="#111111", width=1.0, style=Qt.PenStyle.DashLine),
                    name=f"{lead_name} preview",
                )
            self.overview_plot.plot(time_axis, preview_signal[:, lead_index] + offset, pen=pg.mkPen(color=color, width=1.0))

            if self._view_mode == "stacked":
                label = pg.TextItem(text=lead_name, color=color, anchor=(0, 0.5))
                label.setPos(float(time_axis[0]), offset)
                self.main_plot.addItem(label)

        self.selection_region.setRegion((float(time_axis[0]), min(float(time_axis[-1]), float(time_axis[0]) + 1.0)))
        self.selection_region.show()
        self.overview_region.setRegion((float(time_axis[0]), min(float(time_axis[-1]), float(time_axis[0]) + 10.0)))
        self.overview_region.show()
        self.main_plot.setXRange(float(time_axis[0]), min(float(time_axis[-1]), float(time_axis[0]) + 10.0), padding=0.01)
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
        amplitude = float(self._record.signal[sample_index, lead_index])
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
        offsets = self._compute_offsets(self._preview_signal if self._preview_signal is not None else self._record.signal, visible_indices)
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
        lead_index = self._active_lead if self._view_mode == "single" else self._nearest_visible_lead(0.0)
        stats = compute_selection_stats(self._record.time_axis[mask], self._record.signal[mask, lead_index])
        self.selection_changed.emit(stats)

    def _sync_main_from_overview(self) -> None:
        if self._record is None or not self.overview_region.isVisible():
            return
        start, end = self.overview_region.getRegion()
        self.main_plot.blockSignals(True)
        self.main_plot.setXRange(start, end, padding=0.0)
        self.main_plot.blockSignals(False)

    def _sync_overview_from_main(self) -> None:
        if self._record is None or not self.overview_region.isVisible():
            return
        start, end = self.main_plot.viewRange()[0]
        self.overview_region.blockSignals(True)
        self.overview_region.setRegion((start, end))
        self.overview_region.blockSignals(False)
