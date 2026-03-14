from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QSplitter, QStatusBar, QVBoxLayout, QWidget

from app.gui.controls_panel import ControlsPanel
from app.gui.dialogs import SamplingRateDialog
from app.gui.metadata_panel import MetadataPanel
from app.gui.plot_widget import CursorInfo, ECGPlotWidget
from app.io.loader_factory import LoaderFactory
from app.models.ecg_record import ECGRecord
from app.services.preprocessing import build_preview_signal
from app.services.selection_stats import SelectionStats
from app.services.validation import build_time_axis


@dataclass(slots=True)
class LoadedRecord:
    record: ECGRecord
    preview_signal: np.ndarray


class LoaderSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class LoadFileTask(QRunnable):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path
        self.signals = LoaderSignals()

    def run(self) -> None:
        try:
            loader = LoaderFactory.create_loader(self.file_path)
            record = loader.load(self.file_path)
            preview_signal = build_preview_signal(record.signal, record.sampling_rate, remove_baseline=True, apply_lowpass=True)
            self.signals.finished.emit(LoadedRecord(record=record, preview_signal=preview_signal))
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EKG Viewer - Etap 1")
        self.resize(1600, 900)

        self.thread_pool = QThreadPool.globalInstance()
        self.current_record: ECGRecord | None = None
        self.preview_signal: np.ndarray | None = None

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        central_layout.addWidget(splitter)

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)

        self.controls = ControlsPanel(self)
        self.metadata_panel = MetadataPanel(self)
        left_layout.addWidget(self.controls)
        left_layout.addWidget(self.metadata_panel)
        left_layout.addStretch(1)

        self.plot_widget = ECGPlotWidget(self)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.plot_widget)
        splitter.setSizes([350, 1200])

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.cursor_label = QLabel("Kursor: -")
        self.selection_label = QLabel("Zaznaczenie: -")
        self.status_bar.addPermanentWidget(self.cursor_label, stretch=1)
        self.status_bar.addPermanentWidget(self.selection_label, stretch=2)

        self._connect_signals()

    def _connect_signals(self) -> None:
        self.controls.load_requested.connect(self._choose_file)
        self.controls.view_mode_changed.connect(self.plot_widget.set_view_mode)
        self.controls.active_lead_changed.connect(self.plot_widget.set_active_lead)
        self.controls.lead_visibility_changed.connect(self.plot_widget.set_lead_visibility)
        self.controls.window_preset_selected.connect(self.plot_widget.set_window_seconds)
        self.controls.reset_view_requested.connect(self.plot_widget.reset_view)
        self.controls.grid_toggled.connect(self.plot_widget.set_grid_visible)
        self.controls.raw_toggled.connect(self.plot_widget.set_raw_visible)
        self.controls.filtered_toggled.connect(self.plot_widget.set_filtered_visible)
        self.controls.go_to_start_requested.connect(self.plot_widget.go_to_start)
        self.controls.go_to_end_requested.connect(self.plot_widget.go_to_end)
        self.controls.sampling_rate_changed.connect(self._override_sampling_rate)

        self.plot_widget.cursor_changed.connect(self._update_cursor_status)
        self.plot_widget.selection_changed.connect(self._update_selection_status)

    def _choose_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz plik EKG",
            "",
            "ECG Files (*.hea *.dat *.edf *.csv *.txt *.dcm);;All Files (*)",
        )
        if not file_path:
            return
        self.status_bar.showMessage("Ładowanie pliku...", 5000)
        task = LoadFileTask(file_path)
        task.signals.finished.connect(self._handle_loaded_record)
        task.signals.failed.connect(self._handle_load_error)
        self.thread_pool.start(task)

    def _handle_loaded_record(self, payload: LoadedRecord) -> None:
        record = payload.record
        if record.metadata.get("sampling_rate_defaulted"):
            dialog = SamplingRateDialog(record.sampling_rate, self)
            if dialog.exec():
                record = self._build_record_with_sampling_rate(record, dialog.sampling_rate)
                payload = LoadedRecord(
                    record=record,
                    preview_signal=build_preview_signal(record.signal, record.sampling_rate, remove_baseline=True, apply_lowpass=True),
                )

        self.current_record = payload.record
        self.preview_signal = payload.preview_signal
        self.metadata_panel.set_record(payload.record)
        self.controls.set_leads(payload.record.lead_names)
        self.controls.set_sampling_rate_controls(
            payload.record.sampling_rate,
            enabled=payload.record.source_format == "csv",
        )
        self.plot_widget.set_record(payload.record, payload.preview_signal)
        self.status_bar.showMessage(f"Wczytano {payload.record.file_name}", 5000)

    def _handle_load_error(self, message: str) -> None:
        QMessageBox.critical(self, "Błąd odczytu", message)
        self.status_bar.showMessage("Nie udało się wczytać pliku.", 5000)

    def _override_sampling_rate(self, value: float) -> None:
        if self.current_record is None or self.current_record.source_format != "csv":
            return
        self.current_record = self._build_record_with_sampling_rate(self.current_record, value)
        self.preview_signal = build_preview_signal(
            self.current_record.signal,
            self.current_record.sampling_rate,
            remove_baseline=True,
            apply_lowpass=True,
        )
        self.metadata_panel.set_record(self.current_record)
        self.plot_widget.set_record(self.current_record, self.preview_signal)

    def _build_record_with_sampling_rate(self, record: ECGRecord, sampling_rate: float) -> ECGRecord:
        metadata = record.metadata.copy()
        metadata["sampling_rate_defaulted"] = False
        metadata["sampling_rate_overridden"] = True
        return record.copy_with(
            sampling_rate=sampling_rate,
            time_axis=build_time_axis(record.n_samples, sampling_rate),
            metadata=metadata,
        )

    def _update_cursor_status(self, info: CursorInfo) -> None:
        self.cursor_label.setText(
            f"Kursor: t={info.time_value:.3f} s | próbka={info.sample_index} | amp={info.amplitude:.4f} | lead={info.lead_name}"
        )

    def _update_selection_status(self, stats: SelectionStats) -> None:
        self.selection_label.setText(
            "Zaznaczenie: "
            f"{stats.start_time:.3f}-{stats.end_time:.3f} s | dt={stats.duration:.3f} s | "
            f"min={stats.minimum:.4f} | max={stats.maximum:.4f} | mean={stats.mean:.4f} | std={stats.std:.4f}"
        )
