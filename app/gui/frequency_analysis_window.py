from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.models.ecg_record import ECGRecord
from app.services.frequency_analysis import (
    DEFAULT_MAX_FREQUENCY_HZ,
    DEFAULT_STFT_OVERLAP,
    DEFAULT_STFT_WINDOW,
    PreparedSignalSegment,
    compute_fft_psd,
    compute_stft_spectrogram,
    prepare_signal_segment,
)


@dataclass(slots=True)
class FrequencyAnalysisInput:
    """Snapshot of the current record and optional processed signal."""

    record: ECGRecord
    processed_signal: np.ndarray | None
    filtered_available: bool
    active_lead_index: int = 0


class FrequencyAnalysisDialog(QDialog):
    """Advanced frequency-analysis window for the currently loaded ECG record."""

    def __init__(
        self,
        analysis_input: FrequencyAnalysisInput,
        *,
        visible_range_provider: Callable[[], tuple[float, float] | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analiza czestotliwosciowa")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(1280, 820)

        self._analysis_input = analysis_input
        self._visible_range_provider = visible_range_provider
        self._spectrogram_image = pg.ImageItem(axisOrder="row-major")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        controls_container = QWidget(self)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_container.setMinimumWidth(320)
        controls_container.setMaximumWidth(380)

        controls_layout.addWidget(self._build_signal_group())
        controls_layout.addWidget(self._build_psd_group())
        controls_layout.addWidget(self._build_stft_group())

        self.message_label = QLabel("Wybierz parametry i kliknij Przelicz.", self)
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("color: #444444;")
        controls_layout.addWidget(self.message_label)

        action_layout = QHBoxLayout()
        self.recalculate_button = QPushButton("Przelicz", self)
        self.recalculate_button.clicked.connect(self.recalculate)
        self.save_psd_button = QPushButton("Zapisz wykres PSD", self)
        self.save_psd_button.clicked.connect(lambda: self._export_plot(self.psd_plot, "psd"))
        self.save_spectrogram_button = QPushButton("Zapisz spektrogram", self)
        self.save_spectrogram_button.clicked.connect(lambda: self._export_plot(self.spectrogram_plot, "spectrogram"))
        action_layout.addWidget(self.recalculate_button)
        action_layout.addWidget(self.save_psd_button)
        action_layout.addWidget(self.save_spectrogram_button)
        controls_layout.addLayout(action_layout)
        controls_layout.addStretch(1)

        plots_container = QWidget(self)
        plots_layout = QVBoxLayout(plots_container)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(8)

        self.psd_plot = pg.PlotWidget()
        self.psd_plot.setBackground("w")
        self.psd_plot.setMouseEnabled(x=False, y=False)
        self.psd_plot.showGrid(x=True, y=True, alpha=0.2)
        self.psd_plot.setLabel("bottom", "Czestotliwosc", units="Hz")
        self.psd_plot.setLabel("left", "Gestosc mocy widmowej")
        self.psd_plot.setTitle("PSD przez FFT")

        self.spectrogram_plot = pg.PlotWidget()
        self.spectrogram_plot.setBackground("w")
        self.spectrogram_plot.setMouseEnabled(x=False, y=False)
        self.spectrogram_plot.showGrid(x=True, y=True, alpha=0.15)
        self.spectrogram_plot.setLabel("bottom", "Czas", units="s")
        self.spectrogram_plot.setLabel("left", "Czestotliwosc", units="Hz")
        self.spectrogram_plot.setTitle("Spektrogram STFT")
        self.spectrogram_plot.addItem(self._spectrogram_image)
        self._spectrogram_image.setLookupTable(self._build_lookup_table())

        plots_layout.addWidget(self.psd_plot, stretch=1)
        plots_layout.addWidget(self.spectrogram_plot, stretch=2)

        main_layout.addWidget(controls_container, stretch=0)
        main_layout.addWidget(plots_container, stretch=1)

        self.update_input_data(analysis_input)
        self.recalculate()

    def update_input_data(self, analysis_input: FrequencyAnalysisInput) -> None:
        """Refresh dialog state after loading another record or changing filters."""
        self._analysis_input = analysis_input
        record = analysis_input.record

        self.lead_combo.blockSignals(True)
        self.lead_combo.clear()
        for index, lead_name in enumerate(record.lead_names):
            self.lead_combo.addItem(lead_name, userData=index)
        safe_lead_index = min(max(analysis_input.active_lead_index, 0), max(record.n_leads - 1, 0))
        self.lead_combo.setCurrentIndex(safe_lead_index)
        self.lead_combo.blockSignals(False)

        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("Surowy", userData="raw")
        if analysis_input.filtered_available and analysis_input.processed_signal is not None:
            self.source_combo.addItem("Filtrowany", userData="filtered")
            self.source_combo.setCurrentIndex(1)
        self.source_combo.blockSignals(False)

        full_start = float(record.time_axis[0])
        full_end = float(record.time_axis[-1])
        self.start_edit.setText(f"{full_start:.3f}")
        self.end_edit.setText(f"{full_end:.3f}")
        self.range_whole_radio.setChecked(True)
        self._sync_manual_range_state()

    def recalculate(self) -> None:
        try:
            segment = self._resolve_segment()
        except ValueError as exc:
            self._clear_plots()
            self._set_message(str(exc), error=True)
            return

        signal_source_label = "Filtrowany" if self.source_combo.currentData() == "filtered" else "Surowy"
        lead_name = self.lead_combo.currentText()
        max_frequency_hz = float(self.max_frequency_spin.value())
        requested_nfft = self.nfft_spin.value() if self.nfft_spin.value() > 0 else None

        psd_result = compute_fft_psd(segment.signal, self._analysis_input.record.sampling_rate, max_frequency_hz=max_frequency_hz)
        spectrogram_result = compute_stft_spectrogram(
            segment.signal,
            self._analysis_input.record.sampling_rate,
            segment_start_time=segment.start_time,
            nperseg=self.nperseg_spin.value(),
            noverlap=self.noverlap_spin.value(),
            nfft=requested_nfft,
            max_frequency_hz=max_frequency_hz,
        )

        messages: list[str] = []
        if psd_result.message is not None:
            self.psd_plot.clear()
            self.psd_plot.setTitle("PSD przez FFT")
            messages.append(psd_result.message)
        else:
            self._render_psd(psd_result, lead_name=lead_name, source_label=signal_source_label)

        if spectrogram_result.message is not None:
            self._clear_spectrogram()
            messages.append(spectrogram_result.message)
        else:
            self._render_spectrogram(spectrogram_result)
            if (
                spectrogram_result.nperseg != self.nperseg_spin.value()
                or spectrogram_result.noverlap != self.noverlap_spin.value()
                or spectrogram_result.nfft != (requested_nfft or spectrogram_result.nfft)
            ):
                messages.append("Parametry STFT zostaly automatycznie dopasowane do dlugosci wybranego fragmentu sygnalu.")

        if messages:
            self._set_message(" ".join(messages), error=False)
        else:
            self._set_message(
                f"Przeliczono analize dla odprowadzenia {lead_name} ({signal_source_label.lower()}).",
                error=False,
            )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        parent = self.parent()
        if parent is not None and hasattr(parent, "_frequency_analysis_dialog"):
            parent._frequency_analysis_dialog = None
        super().closeEvent(event)

    def _build_signal_group(self) -> QGroupBox:
        group = QGroupBox("Zakres analizy", self)
        layout = QVBoxLayout(group)
        form = QFormLayout()

        self.lead_combo = QComboBox(self)
        self.source_combo = QComboBox(self)
        form.addRow("Odprowadzenie", self.lead_combo)
        form.addRow("Zrodlo sygnalu", self.source_combo)
        layout.addLayout(form)

        self.range_whole_radio = QRadioButton("Caly sygnal", self)
        self.range_visible_radio = QRadioButton("Aktualnie widoczne okno", self)
        self.range_manual_radio = QRadioButton("Reczny zakres", self)
        self.range_whole_radio.setChecked(True)
        for radio in (self.range_whole_radio, self.range_visible_radio, self.range_manual_radio):
            radio.toggled.connect(self._sync_manual_range_state)
            layout.addWidget(radio)

        manual_form = QFormLayout()
        self.start_edit = QLineEdit(self)
        self.end_edit = QLineEdit(self)
        self.start_edit.setPlaceholderText("np. 0.000")
        self.end_edit.setPlaceholderText("np. 5.000")
        manual_form.addRow("t_start [s]", self.start_edit)
        manual_form.addRow("t_end [s]", self.end_edit)
        layout.addLayout(manual_form)
        return group

    def _build_psd_group(self) -> QGroupBox:
        group = QGroupBox("PSD przez FFT", self)
        form = QFormLayout(group)

        self.max_frequency_spin = QSpinBox(self)
        self.max_frequency_spin.setRange(1, 1000)
        self.max_frequency_spin.setValue(int(DEFAULT_MAX_FREQUENCY_HZ))

        self.psd_scale_combo = QComboBox(self)
        self.psd_scale_combo.addItem("Liniowa", userData="linear")
        self.psd_scale_combo.addItem("Logarytmiczna", userData="log")

        form.addRow("Max czestotliwosc [Hz]", self.max_frequency_spin)
        form.addRow("Skala osi Y", self.psd_scale_combo)
        return group

    def _build_stft_group(self) -> QGroupBox:
        group = QGroupBox("Spektrogram STFT", self)
        form = QFormLayout(group)

        self.nperseg_spin = QSpinBox(self)
        self.nperseg_spin.setRange(16, 8192)
        self.nperseg_spin.setValue(DEFAULT_STFT_WINDOW)

        self.noverlap_spin = QSpinBox(self)
        self.noverlap_spin.setRange(0, 8191)
        self.noverlap_spin.setValue(DEFAULT_STFT_OVERLAP)

        self.nfft_spin = QSpinBox(self)
        self.nfft_spin.setRange(0, 16384)
        self.nfft_spin.setSpecialValueText("Auto")
        self.nfft_spin.setValue(0)

        form.addRow("Dlugosc okna", self.nperseg_spin)
        form.addRow("Overlap", self.noverlap_spin)
        form.addRow("FFT size", self.nfft_spin)
        return group

    def _sync_manual_range_state(self) -> None:
        manual_enabled = self.range_manual_radio.isChecked()
        self.start_edit.setEnabled(manual_enabled)
        self.end_edit.setEnabled(manual_enabled)

    def _resolve_segment(self) -> PreparedSignalSegment:
        record = self._analysis_input.record
        lead_index = int(self.lead_combo.currentData())
        source_key = str(self.source_combo.currentData())

        if source_key == "filtered":
            if self._analysis_input.processed_signal is None or not self._analysis_input.filtered_available:
                raise ValueError("Sygnal filtrowany nie jest aktualnie dostepny.")
            signal_matrix = self._analysis_input.processed_signal
        else:
            signal_matrix = record.signal

        lead_signal = np.asarray(signal_matrix[:, lead_index], dtype=float)
        start_time: float | None = None
        end_time: float | None = None

        if self.range_visible_radio.isChecked():
            visible_range = self._visible_range_provider()
            if visible_range is None:
                raise ValueError("Nie mozna odczytac aktualnie widocznego zakresu czasu.")
            start_time, end_time = visible_range
        elif self.range_manual_radio.isChecked():
            try:
                start_time = float(self.start_edit.text().replace(",", "."))
                end_time = float(self.end_edit.text().replace(",", "."))
            except ValueError as exc:
                raise ValueError("Podaj poprawny reczny zakres czasu w sekundach.") from exc

        return prepare_signal_segment(
            lead_signal,
            record.time_axis,
            start_time=start_time,
            end_time=end_time,
        )

    def _render_psd(self, result, *, lead_name: str, source_label: str) -> None:
        self.psd_plot.clear()
        self.psd_plot.showGrid(x=True, y=True, alpha=0.2)
        self.psd_plot.setLabel("bottom", "Czestotliwosc", units="Hz")
        self.psd_plot.setLabel("left", result.y_label)
        self.psd_plot.setTitle(f"PSD przez FFT: {lead_name} ({source_label.lower()})")

        frequencies = np.asarray(result.frequencies_hz, dtype=float)
        values = np.asarray(result.power, dtype=float)
        if self.psd_scale_combo.currentData() == "log":
            values = np.maximum(values, np.finfo(float).tiny)
            self.psd_plot.setLogMode(x=False, y=True)
        else:
            self.psd_plot.setLogMode(x=False, y=False)

        if frequencies.size > 1:
            bar_width = max(float(np.min(np.diff(frequencies))) * 0.85, 1e-6)
        else:
            bar_width = max(float(self.max_frequency_spin.value()) * 0.02, 0.1)
        bars = pg.BarGraphItem(
            x=frequencies,
            height=values,
            width=bar_width,
            brush=pg.mkBrush("#00429d"),
            pen=pg.mkPen("#00429d"),
        )
        self.psd_plot.addItem(bars)
        self.psd_plot.setXRange(0.0, float(self.max_frequency_spin.value()), padding=0.01)

    def _render_spectrogram(self, result) -> None:
        self.spectrogram_plot.clear()
        self.spectrogram_plot.addItem(self._spectrogram_image)
        self.spectrogram_plot.showGrid(x=True, y=True, alpha=0.15)
        self.spectrogram_plot.setLabel("bottom", "Czas", units="s")
        self.spectrogram_plot.setLabel("left", "Czestotliwosc", units="Hz")
        self.spectrogram_plot.setTitle("Spektrogram STFT")

        image = np.asarray(result.power_db, dtype=float)
        self._spectrogram_image.setLookupTable(self._build_lookup_table())
        self._spectrogram_image.setImage(image, autoLevels=True)

        x_min = float(result.times_s[0])
        x_max = float(result.times_s[-1]) if result.times_s.size > 1 else x_min + 1.0
        y_min = float(result.frequencies_hz[0])
        y_max = float(result.frequencies_hz[-1]) if result.frequencies_hz.size > 1 else y_min + 1.0
        width = max(x_max - x_min, 1e-6)
        height = max(y_max - y_min, 1e-6)
        self._spectrogram_image.setRect(QRectF(x_min, y_min, width, height))
        self.spectrogram_plot.setXRange(x_min, x_max, padding=0.01)
        self.spectrogram_plot.setYRange(y_min, y_max, padding=0.01)

    def _clear_plots(self) -> None:
        self.psd_plot.clear()
        self._clear_spectrogram()

    def _clear_spectrogram(self) -> None:
        self.spectrogram_plot.clear()
        self.spectrogram_plot.addItem(self._spectrogram_image)
        self._spectrogram_image.setImage(np.empty((0, 0)), autoLevels=False)
        self.spectrogram_plot.setTitle("Spektrogram STFT")

    def _set_message(self, message: str, *, error: bool) -> None:
        self.message_label.setText(message)
        color = "#B00020" if error else "#444444"
        self.message_label.setStyleSheet(f"color: {color};")

    def _export_plot(self, plot_widget: pg.PlotWidget, prefix: str) -> None:
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Zapisz obraz",
            str(Path.home() / f"{prefix}.png"),
            "PNG (*.png)",
        )
        if not file_name:
            return
        try:
            exporter = ImageExporter(plot_widget.plotItem)
            exporter.export(file_name)
            self._set_message(f"Zapisano obraz: {file_name}", error=False)
        except Exception as exc:  # pragma: no cover - GUI export failure path
            QMessageBox.critical(self, "Blad zapisu", str(exc))

    @staticmethod
    def _build_lookup_table() -> np.ndarray:
        color_map = pg.ColorMap(
            pos=np.array([0.0, 0.2, 0.5, 0.8, 1.0]),
            color=np.array(
                [
                    (12, 7, 134, 255),
                    (63, 81, 181, 255),
                    (33, 145, 140, 255),
                    (144, 190, 109, 255),
                    (249, 231, 33, 255),
                ],
                dtype=np.ubyte,
            ),
        )
        return color_map.getLookupTable(0.0, 1.0, 256)
