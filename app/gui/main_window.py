from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QStackedWidget,
)

from app.gui.controls_panel import ControlsPanel
from app.gui.dialogs import (
    DiseaseResultDialog,
    MetadataDialog,
    SamplingRateDialog,
    WaitPopupDialog,
)
from app.gui.analysis_tab import FrequencyAnalysisDialog, FrequencyAnalysisInput
from app.gui.plot_widget import CursorInfo, ECGPlotWidget
from app.io.store_factory import StoreFactory
from app.models.ecg_record import ECGRecord
from app.services.preprocessing import (
    SignalFilterConfig,
    default_filter_config,
    preprocess_signal,
)
from app.services.selection_stats import SelectionStats
from app.services.validation import build_time_axis
from disease_detector import DiseaseDetector


PLAYBACK_TIMER_INTERVAL_MS = 50
PLAYBACK_FALLBACK_WINDOW_SECONDS = 10.0


@dataclass(slots=True)
class LoadedRecord:
    record: ECGRecord


@dataclass(slots=True)
class PlaybackState:
    is_playing: bool = False
    is_paused: bool = False
    current_time_sec: float = 0.0
    playback_speed: float = 1.0
    loop_enabled: bool = False


class LoaderSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class DiseaseDetectionSignals(QObject):
    finished = Signal(dict)
    failed = Signal(str)


class LoadFileTask(QRunnable):
    """Background loader task used to keep the GUI responsive."""

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path
        self.signals = LoaderSignals()

    def run(self) -> None:
        try:
            loader = StoreFactory.create_loader(self.file_path)
            record = loader.load(self.file_path)
            self.signals.finished.emit(LoadedRecord(record=record))
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class DiseaseDetectionTask(QRunnable):
    def __init__(self, signal: np.ndarray, sampling_rate: float, models_dir: Path) -> None:
        super().__init__()
        self.signal = np.asarray(signal, dtype=np.float32)
        self.sampling_rate = float(sampling_rate)
        self.models_dir = models_dir
        self.signals = DiseaseDetectionSignals()

    def run(self) -> None:
        try:
            detector = DiseaseDetector(self.models_dir)
            detector.load_models()
            self.signals.finished.emit(
                detector.predict(self.signal, fs=self.sampling_rate)
            )
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class MainWindow(QMainWindow):
    """Main Stage 1 ECG import and visualization window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EKG Viewer")

        self.thread_pool = QThreadPool.globalInstance()
        self.current_record: ECGRecord | None = None
        self.processed_signal: np.ndarray | None = None
        self._frequency_analysis_panel: FrequencyAnalysisDialog | None = None
        self._preload_ui_active = True
        self._wait_popup_depth = 0
        self._wait_dialog: WaitPopupDialog | None = None
        self.filter_config: SignalFilterConfig = default_filter_config()
        self.playback_state = PlaybackState()
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(PLAYBACK_TIMER_INTERVAL_MS)
        self.playback_timer.timeout.connect(self._advance_playback)
        self._build_menu()

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(6, 6, 6, 6)
        central_layout.setSpacing(6)

        self.controls = ControlsPanel(self)
        self.metadata_dialog = MetadataDialog(self)

        self.sidebar_stack = QStackedWidget(self)

        ecg_sidebar = QWidget(self)
        ecg_sidebar_layout = QVBoxLayout(ecg_sidebar)
        ecg_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        ecg_sidebar_layout.setSpacing(8)
        ecg_sidebar_layout.addWidget(self.controls)
        ecg_sidebar_layout.addStretch(1)

        self.analysis_sidebar = QWidget(self)
        self.analysis_sidebar_layout = QVBoxLayout(self.analysis_sidebar)
        self.analysis_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_sidebar_layout.setSpacing(8)
        self.analysis_sidebar_layout.addStretch(1)

        self.sidebar_stack.addWidget(ecg_sidebar)
        self.sidebar_stack.addWidget(self.analysis_sidebar)
        self.sidebar_stack.setFixedWidth(360)
        self.sidebar_stack.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )

        left_scroll = QScrollArea(self)
        left_scroll.setFixedWidth(360)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(self.sidebar_stack)

        self.plot_widget = ECGPlotWidget(self)
        self.plot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.content_tabs = QTabWidget(self)
        self.content_tabs.setDocumentMode(True)

        self.ecg_tab = QWidget(self)
        ecg_tab_layout = QVBoxLayout(self.ecg_tab)
        ecg_tab_layout.setContentsMargins(0, 0, 0, 0)
        ecg_tab_layout.setSpacing(0)
        ecg_tab_layout.addWidget(self.plot_widget)

        self.analysis_tab = QWidget(self)
        self.analysis_tab_layout = QVBoxLayout(self.analysis_tab)
        self.analysis_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_tab_layout.setSpacing(0)
        self.analysis_placeholder = QLabel(
            "Wczytaj plik EKG, aby otworzyc analize.", self.analysis_tab
        )
        self.analysis_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analysis_placeholder.setStyleSheet("color: #666666;")
        self.analysis_tab_layout.addWidget(self.analysis_placeholder)

        self.ecg_tab_index = self.content_tabs.addTab(self.ecg_tab, "EKG")
        self.analysis_tab_index = self.content_tabs.addTab(self.analysis_tab, "Analiza")
        self.info_button = QPushButton("Informacje", self)
        self.info_button.setEnabled(False)
        self.info_button.clicked.connect(self._open_metadata_dialog)
        self.content_tabs.setCornerWidget(
            self.info_button, Qt.Corner.TopRightCorner
        )

        central_layout.addWidget(left_scroll)
        central_layout.addWidget(self.content_tabs, stretch=1)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.file_info_label = QLabel("Plik: -")
        self.playback_status_label = QLabel("Odtwarzanie: zatrzymane")
        self.cursor_label = QLabel("Kursor: -")
        self.selection_label = QLabel("Zaznaczenie techniczne: -")
        self.selection_label.setToolTip(
            "Techniczne statystyki dla odprowadzenia wynikajacego z biezacego widoku "
            "i zaznaczenia na wykresie. Bez adnotacji klinicznych, "
            "detekcji zalamkow i analizy wieloodprowadzeniowej."
        )
        self.status_bar.addWidget(self.file_info_label, stretch=2)
        self.status_bar.addPermanentWidget(self.playback_status_label, stretch=1)
        self.status_bar.addPermanentWidget(self.cursor_label, stretch=1)
        self.status_bar.addPermanentWidget(self.selection_label, stretch=2)

        self._connect_signals()
        self._apply_screen_adaptive_geometry()
        self.controls.sync_signal_display_mode(filters_active=False)
        self._update_frequency_analysis_action_state()
        self._sync_sidebar_to_current_tab(self.content_tabs.currentIndex())
        self._set_preload_ui_state(True)

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Plik")
        self.open_file_action = QAction("Wczytaj plik", self)
        self.open_file_action.setStatusTip(
            "Wczytaj zapis EKG albo zapisany fragment z pliku."
        )
        self.open_file_action.triggered.connect(self._choose_file)
        file_menu.addAction(self.open_file_action)

        self.save_fragment_action = QAction("Zapisz zaznaczony fragment", self)
        self.save_fragment_action.setStatusTip(
            "Zapisz aktualnie zaznaczony fragment EKG w wybranym formacie."
        )
        self.save_fragment_action.setEnabled(False)
        self.save_fragment_action.triggered.connect(self._save_selected_fragment)

    def _apply_screen_adaptive_geometry(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 900)
            return

        available = screen.availableGeometry()
        width = max(1200, min(int(available.width() * 0.88), 2200))
        height = max(760, min(int(available.height() * 0.88), 1400))
        self.resize(width, height)
        self.move(
            available.x() + max((available.width() - width) // 2, 0),
            available.y() + max((available.height() - height) // 2, 0),
        )

    def _connect_signals(self) -> None:
        self.content_tabs.currentChanged.connect(self._sync_sidebar_to_current_tab)
        self.plot_widget.load_requested.connect(self._choose_file)
        self.controls.lead_visibility_changed.connect(self._set_lead_visibility_with_wait)
        self.controls.window_preset_selected.connect(self._set_window_preset)
        self.controls.reset_view_requested.connect(self._reset_view_with_wait)
        self.controls.grid_toggled.connect(self.plot_widget.set_grid_visible)
        self.controls.raw_toggled.connect(self._set_raw_visible_with_wait)
        self.controls.filtered_toggled.connect(self._set_filtered_visible_with_wait)
        self.controls.sampling_rate_changed.connect(self._override_sampling_rate)
        self.controls.filter_config_changed.connect(self._apply_filter_config)
        self.controls.disease_detection_requested.connect(self._detect_diseases)

        self.plot_widget.cursor_changed.connect(self._update_cursor_status)
        self.plot_widget.selection_changed.connect(self._update_selection_status)
        self.plot_widget.selection_context_menu_requested.connect(
            self._open_selection_context_menu
        )
        self.plot_widget.play_requested.connect(self._play)
        self.plot_widget.pause_requested.connect(self._pause)
        self.plot_widget.step_backward_requested.connect(self._step_backward)
        self.plot_widget.stop_requested.connect(self._stop)
        self.plot_widget.step_forward_requested.connect(self._step_forward)
        self.plot_widget.playback_speed_changed.connect(self._set_playback_speed)
        self.plot_widget.playback_loop_toggled.connect(self._set_playback_loop)
        self.plot_widget.playback_position_changed.connect(self._seek_playback_fraction)
        self.plot_widget.visible_time_range_changed.connect(
            self._refresh_frequency_analysis_for_visible_range
        )

    @staticmethod
    def _sampling_rate_control_state(record: ECGRecord) -> tuple[bool, str]:
        if record.source_format != "csv":
            return (
                False,
                "Sampling rate comes from the source file for WFDB, EDF and DICOM. "
                "Manual override is intended mainly for CSV/TXT without an explicit time axis.",
            )
        if record.metadata.get("time_axis_source") == "file":
            return (
                False,
                "This CSV/TXT file contains an explicit time axis, so sampling rate is inferred from it. "
                "Manual override is intentionally limited in this case.",
            )
        return (
            True,
            "Use this mainly for CSV/TXT without an explicit time axis. "
            "Changing the value rebuilds only the generated time axis for the current import.",
        )

    def _choose_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz plik EKG",
            "",
            "ECG Files (*.hea *.dat *.edf *.csv *.txt *.dcm *.atr);;All Files (*)",
        )
        if not file_path:
            return
        self.status_bar.showMessage("Ladowanie pliku...", 5000)
        self._show_wait_popup()
        task = LoadFileTask(file_path)
        task.signals.finished.connect(self._handle_loaded_record)
        task.signals.failed.connect(self._handle_load_error)
        self.thread_pool.start(task)

    def _handle_loaded_record(self, payload: LoadedRecord) -> None:
        record = payload.record
        if record.metadata.get("sampling_rate_defaulted"):
            dialog = SamplingRateDialog(record.sampling_rate, self)
            if dialog.exec():
                record = self._build_record_with_sampling_rate(
                    record, dialog.sampling_rate
                )

        self.current_record = record
        self._reset_playback()
        self.metadata_dialog.set_record(record)
        self._update_file_info(record)
        self.controls.set_leads(record.lead_names)
        sampling_rate_enabled, sampling_rate_tooltip = (
            self._sampling_rate_control_state(record)
        )
        self.controls.set_sampling_rate_controls(
            record.sampling_rate,
            enabled=sampling_rate_enabled,
            tooltip=sampling_rate_tooltip,
        )
        self._refresh_processed_signal()
        self._refresh_frequency_analysis_dialog()
        self.selection_label.setText("Zaznaczenie techniczne: -")
        self._update_fragment_action_state()
        self._set_preload_ui_state(False)
        self.status_bar.showMessage(f"Wczytano {record.file_name}", 5000)
        QApplication.processEvents()
        self._close_wait_popup()

    def _handle_load_error(self, message: str) -> None:
        self._set_preload_ui_state(True)
        self._close_wait_popup()
        QMessageBox.critical(self, "Blad odczytu", message)
        self.status_bar.showMessage("Nie udalo sie wczytac pliku.", 5000)

    def _override_sampling_rate(self, value: float) -> None:
        if self.current_record is None:
            return
        enabled, tooltip = self._sampling_rate_control_state(self.current_record)
        if not enabled:
            self.status_bar.showMessage(tooltip, 7000)
            return
        def update_sampling_rate() -> None:
            self.current_record = self._build_record_with_sampling_rate(
                self.current_record, value
            )
            self.metadata_dialog.set_record(self.current_record)
            self.controls.set_sampling_rate_controls(
                self.current_record.sampling_rate,
                enabled=True,
                tooltip=self._sampling_rate_control_state(self.current_record)[1],
            )
            self._refresh_processed_signal()

        self._run_with_wait_popup(update_sampling_rate)
        self.status_bar.showMessage(
            "Zaktualizowano sampling rate dla tabelarycznego CSV/TXT bez osi czasu.",
            5000,
        )

    def _apply_filter_config(self, config: SignalFilterConfig) -> None:
        def apply_filter() -> None:
            self.filter_config = config
            self._refresh_processed_signal()

        self._run_with_wait_popup(apply_filter)

    def _refresh_processed_signal(self) -> None:
        if self.current_record is None:
            self.processed_signal = None
            self._update_file_info(None)
            self._update_info_button_state()
            self.controls.set_disease_detection_enabled(False)
            self.plot_widget.set_playback_enabled(False)
            self.plot_widget.set_playback_position(0.0, 0.0)
            self._set_playback_status("zatrzymane")
            self.plot_widget.set_record(None, None, filtering_active=False)
            self._update_fragment_action_state()
            self._update_frequency_analysis_action_state()
            return
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            self.processed_signal = preprocess_signal(
                self.current_record.signal,
                self.current_record.sampling_rate,
                self.filter_config,
            )
        self.plot_widget.set_record(
            self.current_record,
            self.processed_signal,
            filtering_active=self.filter_config.any_enabled(),
        )
        self.controls.set_disease_detection_enabled(self.current_record is not None)
        self.plot_widget.set_playback_enabled(self._playback_available())
        self.controls.sync_signal_display_mode(
            filters_active=self.filter_config.any_enabled()
        )
        self._update_info_button_state()
        self._render_current_window()
        self._update_playback_position_display()
        self._update_fragment_action_state()
        self._update_frequency_analysis_action_state()
        if captured_warnings:
            self.status_bar.showMessage(str(captured_warnings[-1].message), 7000)

    def _open_metadata_dialog(self) -> None:
        self.metadata_dialog.set_record(self.current_record)
        self.metadata_dialog.exec()

    def _detect_diseases(self) -> None:
        if self.current_record is None:
            QMessageBox.information(self, "Brak danych", "Najpierw wczytaj plik EKG.")
            return

        models_dir = Path(__file__).resolve().parents[2] / "models"
        available_models = (
            models_dir / "rf_model.pkl",
            models_dir / "svm_model.pkl",
            models_dir / "cnn_model.pth",
        )
        if not models_dir.exists() or not any(path.exists() for path in available_models):
            QMessageBox.information(
                self,
                "Brak modeli",
                "Brak wytrenowanych modeli. Uruchom skrypt save_models.py po treningu w notatniku Jupyter.",
            )
            return

        task = DiseaseDetectionTask(
            signal=self._active_signal_for_disease_detection(),
            sampling_rate=self.current_record.sampling_rate,
            models_dir=models_dir,
        )
        task.signals.finished.connect(self._handle_disease_detection_finished)
        task.signals.failed.connect(self._handle_disease_detection_failed)
        self._show_disease_progress()
        self.thread_pool.start(task)

    def _active_signal_for_disease_detection(self) -> np.ndarray:
        if self.current_record is None:
            raise RuntimeError("No ECG record loaded.")
        lead_index = self.controls.primary_selected_lead_index()
        return np.asarray(self.current_record.signal[:, lead_index], dtype=np.float32)

    def _show_disease_progress(self) -> None:
        self._show_wait_popup()

    def _close_disease_progress(self) -> None:
        self._close_wait_popup()

    def _show_wait_popup(self) -> None:
        self._wait_popup_depth += 1
        if self._wait_popup_depth > 1:
            return
        if self._wait_dialog is None:
            self._wait_dialog = WaitPopupDialog(self)
        self._wait_dialog.show()
        self._wait_dialog.raise_()
        self._wait_dialog.activateWindow()
        if QGuiApplication.overrideCursor() is None:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _close_wait_popup(self) -> None:
        if self._wait_popup_depth > 0:
            self._wait_popup_depth -= 1
        if self._wait_popup_depth > 0:
            return
        if self._wait_dialog is not None:
            self._wait_dialog.hide()
        if QGuiApplication.overrideCursor() is not None:
            QGuiApplication.restoreOverrideCursor()

    def _run_with_wait_popup(self, operation) -> None:
        self._show_wait_popup()
        QApplication.processEvents()
        try:
            operation()
            QApplication.processEvents()
        finally:
            self._close_wait_popup()

    def _set_lead_visibility_with_wait(self, visibility: dict[int, bool]) -> None:
        self._run_with_wait_popup(
            lambda: self.plot_widget.set_lead_visibility(visibility)
        )

    def _reset_view_with_wait(self) -> None:
        self._run_with_wait_popup(self.plot_widget.reset_view)

    def _set_raw_visible_with_wait(self, visible: bool) -> None:
        self._run_with_wait_popup(lambda: self.plot_widget.set_raw_visible(visible))

    def _set_filtered_visible_with_wait(self, visible: bool) -> None:
        self._run_with_wait_popup(
            lambda: self.plot_widget.set_filtered_visible(visible)
        )

    def _set_preload_ui_state(self, preload: bool) -> None:
        self._preload_ui_active = preload
        has_record = self.current_record is not None
        self.controls.setEnabled(not preload)
        self.plot_widget.set_preload_state(preload)
        self.menuBar().setEnabled(not preload)
        self.open_file_action.setEnabled(not preload)
        self.content_tabs.setTabEnabled(self.analysis_tab_index, not preload and has_record)
        self.content_tabs.tabBar().setEnabled(not preload)
        self.info_button.setEnabled(not preload and has_record)
        if preload and self.content_tabs.currentIndex() == self.analysis_tab_index:
            self.content_tabs.setCurrentIndex(self.ecg_tab_index)
        self._update_fragment_action_state()
        self._sync_sidebar_to_current_tab(self.content_tabs.currentIndex())

    def _handle_disease_detection_finished(self, result: dict) -> None:
        self._close_disease_progress()
        self.status_bar.showMessage("Zakonczono analize schorzen EKG.", 5000)
        DiseaseResultDialog(result, self).exec()

    def _handle_disease_detection_failed(self, message: str) -> None:
        self._close_disease_progress()
        QMessageBox.critical(self, "Blad analizy", message)
        self.status_bar.showMessage("Analiza schorzen nie powiodla sie.", 5000)

    def _update_info_button_state(self) -> None:
        self.info_button.setEnabled(
            self.current_record is not None and not self._preload_ui_active
        )

    def _build_record_with_sampling_rate(
        self, record: ECGRecord, sampling_rate: float
    ) -> ECGRecord:
        metadata = record.metadata.copy()
        metadata["sampling_rate_defaulted"] = False
        metadata["sampling_rate_overridden"] = True
        metadata["sampling_rate_note"] = (
            "Sampling rate was manually overridden for a CSV/TXT import without an explicit time axis."
        )
        return record.copy_with(
            sampling_rate=sampling_rate,
            time_axis=build_time_axis(record.n_samples, sampling_rate),
            metadata=metadata,
        )

    def _update_cursor_status(self, info: CursorInfo) -> None:
        self.cursor_label.setText(
            f"Kursor: t={info.time_value:.3f} s | probka={info.sample_index} | amp={info.amplitude:.4f} | lead={info.lead_name}"
        )

    def _update_selection_status(self, stats: SelectionStats | None) -> None:
        if stats is None:
            self.selection_label.setText("Zaznaczenie techniczne: -")
            self._update_fragment_action_state()
            return
        self.selection_label.setText(
            "Zaznaczenie techniczne: "
            f"{stats.start_time:.3f}-{stats.end_time:.3f} s | dt={stats.duration:.3f} s | "
            f"min={stats.minimum:.4f} | max={stats.maximum:.4f} | mean={stats.mean:.4f} | std={stats.std:.4f}"
        )
        self._update_fragment_action_state()

    def _update_file_info(self, record: ECGRecord | None) -> None:
        if record is None:
            self.file_info_label.setText("Plik: -")
            return
        self.file_info_label.setText(
            f"Plik: {record.file_name} | format: {record.source_format.upper()} | "
            f"fs: {record.sampling_rate:.2f} Hz | odprowadzenia: {record.n_leads}"
        )

    def _set_window_preset(self, seconds: int | None) -> None:
        def update_window() -> None:
            self.plot_widget.set_window_seconds(seconds)
            self._render_current_window()
            self._update_playback_position_display()

        self._run_with_wait_popup(update_window)

    def _play(self) -> None:
        if not self._playback_available():
            return
        if self.playback_state.current_time_sec >= self._max_playback_start_time():
            self.playback_state.current_time_sec = 0.0
        self.playback_state.is_playing = True
        self.playback_state.is_paused = False
        if not self.playback_timer.isActive():
            self.playback_timer.start()
        self._set_playback_status("odtwarzanie")
        self._render_current_window()

    def _pause(self) -> None:
        if not self._playback_available():
            return
        self.playback_state.is_playing = False
        self.playback_state.is_paused = True
        self.playback_timer.stop()
        self._set_playback_status("pauza")

    def _stop(self) -> None:
        self.playback_timer.stop()
        self.playback_state.is_playing = False
        self.playback_state.is_paused = False
        self.playback_state.current_time_sec = 0.0
        self._set_playback_status("zatrzymane")
        self._render_current_window()
        self._update_playback_position_display()

    def _step_backward(self) -> None:
        self._step_playback(-self._effective_playback_window_seconds())

    def _step_forward(self) -> None:
        self._step_playback(self._effective_playback_window_seconds())

    def _reset_playback(self) -> None:
        self.playback_timer.stop()
        self.playback_state = PlaybackState(
            playback_speed=self.playback_state.playback_speed,
            loop_enabled=self.playback_state.loop_enabled,
        )
        self._set_playback_status("zatrzymane")

    def _set_playback_speed(self, playback_speed: float) -> None:
        self.playback_state.playback_speed = playback_speed

    def _set_playback_loop(self, enabled: bool) -> None:
        self.playback_state.loop_enabled = enabled

    def _seek_playback_fraction(self, position_fraction: float) -> None:
        if not self._playback_available():
            return
        self.playback_state.current_time_sec = self._max_playback_start_time() * max(
            0.0, min(position_fraction, 1.0)
        )
        self._render_current_window()
        self._update_playback_position_display()

    def _advance_playback(self) -> None:
        if not self.playback_state.is_playing or not self._playback_available():
            self.playback_timer.stop()
            return
        delta_seconds = (
            self.playback_timer.interval() / 5000.0
        ) * self.playback_state.playback_speed
        max_start = self._max_playback_start_time()
        next_time = self.playback_state.current_time_sec + delta_seconds
        if next_time >= max_start:
            if self.playback_state.loop_enabled and max_start > 0:
                next_time = 0.0
            else:
                self.playback_state.current_time_sec = max_start
                self._render_current_window()
                self._update_playback_position_display()
                self._stop()
                return
        self.playback_state.current_time_sec = next_time
        self._render_current_window()
        self._update_playback_position_display()

    def _step_playback(self, delta_seconds: float) -> None:
        if not self._playback_available():
            return
        next_time = self.playback_state.current_time_sec + float(delta_seconds)
        self.playback_state.current_time_sec = min(
            max(next_time, 0.0), self._max_playback_start_time()
        )
        self._render_current_window()
        self._update_playback_position_display()

    def _render_current_window(self) -> None:
        if self.current_record is None:
            return
        self.plot_widget.set_visible_time_window(
            self.playback_state.current_time_sec,
            self._effective_playback_window_seconds(),
        )
        self._refresh_frequency_analysis_for_visible_range()

    def _update_playback_position_display(self) -> None:
        self.plot_widget.set_playback_position(
            self.playback_state.current_time_sec, self._playback_duration_seconds()
        )

    def _playback_available(self) -> bool:
        if self.current_record is None:
            return False
        if (
            self.current_record.n_samples <= 1
            or self.current_record.duration_seconds <= 0
        ):
            return False
        return self.current_record.sampling_rate > 0

    def _playback_duration_seconds(self) -> float:
        if self.current_record is None:
            return 0.0
        return max(float(self.current_record.duration_seconds), 0.0)

    def _effective_playback_window_seconds(self) -> float:
        if self.current_record is None:
            return PLAYBACK_FALLBACK_WINDOW_SECONDS
        selected_window = self.plot_widget.current_window_seconds()
        
        return min(float(selected_window), max(self._playback_duration_seconds(), 0.1))

    def _max_playback_start_time(self) -> float:
        return max(self._playback_duration_seconds(), 0.0)

    def _set_playback_status(self, status_text: str) -> None:
        self.playback_status_label.setText(f"Odtwarzanie: {status_text}")
        self.plot_widget.set_playback_state(status_text.capitalize())

    def _update_frequency_analysis_action_state(self) -> None:
        analysis_available = self.current_record is not None and not self._preload_ui_active
        self.content_tabs.setTabEnabled(self.analysis_tab_index, analysis_available)
        if (
            not analysis_available
            and self.content_tabs.currentIndex() == self.analysis_tab_index
        ):
            self.content_tabs.setCurrentIndex(self.ecg_tab_index)
        self._sync_sidebar_to_current_tab(self.content_tabs.currentIndex())

    def _update_fragment_action_state(self) -> None:
        self.save_fragment_action.setEnabled(
            not self._preload_ui_active
            and self.current_record is not None
            and self.plot_widget.selected_sample_range() is not None
        )

    def _save_selected_fragment(self) -> None:
        if self.current_record is None:
            QMessageBox.information(self, "Brak danych", "Najpierw wczytaj plik EKG.")
            return

        sample_range = self.plot_widget.selected_sample_range()
        time_range = self.plot_widget.selected_time_range()
        if sample_range is None or time_range is None:
            QMessageBox.information(
                self, "Brak zaznaczenia", "Kliknij wykres, aby zaznaczyc fragment EKG."
            )
            return

        try:
            fragment = self.current_record.slice_samples(*sample_range)
        except ValueError as exc:
            QMessageBox.warning(self, "Nieprawidlowe zaznaczenie", str(exc))
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Zapisz zaznaczony fragment",
            str(
                Path(self.current_record.file_path).resolve().parent
                / self._default_fragment_file_name(*time_range)
            ),
            ";;".join(StoreFactory.save_filters()),
            StoreFactory.default_save_filter(self.current_record.source_format),
        )
        if not file_path:
            return

        store, target_path = StoreFactory.resolve_save_target(
            file_path,
            selected_filter,
            fallback_format=self.current_record.source_format,
            original_file_path=self.current_record.file_path,
        )
        saved_path = store.save(fragment, target_path)
        self.status_bar.showMessage(
            f"Zapisano fragment do {Path(saved_path).name}", 5000
        )

    def _open_selection_context_menu(self) -> None:
        if self.plot_widget.selected_sample_range() is None:
            return

        menu = QMenu(self)
        menu.addAction(self.save_fragment_action)
        clear_action = menu.addAction("Wyczysc zaznaczenie")
        chosen_action = menu.exec(self.cursor().pos())
        if chosen_action == clear_action:
            self.plot_widget.clear_selection()

    def _default_fragment_file_name(self, start_time: float, end_time: float) -> str:
        if self.current_record is None:
            return "ecg_fragment.csv"
        stem = Path(self.current_record.file_path).stem
        extension = StoreFactory.preferred_save_extension(
            self.current_record.source_format,
            self.current_record.file_path,
        )
        start_label = self._format_fragment_time_for_file_name(start_time)
        end_label = self._format_fragment_time_for_file_name(end_time)
        return f"{stem}_fragment_{start_label}_{end_label}{extension}"

    @staticmethod
    def _format_fragment_time_for_file_name(time_value: float) -> str:
        milliseconds = int(round(max(time_value, 0.0) * 1000.0))
        seconds, millis = divmod(milliseconds, 1000)
        return f"{seconds}s{millis:03d}ms"

    def _open_frequency_analysis_tab(self) -> None:
        if self.current_record is None:
            QMessageBox.information(self, "Brak danych", "Najpierw wczytaj plik EKG.")
            return

        self._refresh_frequency_analysis_dialog()
        self.content_tabs.setCurrentIndex(self.analysis_tab_index)

    def _refresh_frequency_analysis_dialog(self) -> None:
        if self.current_record is None:
            return
        panel = self._ensure_frequency_analysis_panel()
        panel.update_input_data(self._build_frequency_analysis_input())
        panel.recalculate()

    def _refresh_frequency_analysis_for_visible_range(self) -> None:
        if self._frequency_analysis_panel is None:
            return
        self._frequency_analysis_panel.refresh_for_visible_range_change()

    def _ensure_frequency_analysis_panel(self) -> FrequencyAnalysisDialog:
        if self._frequency_analysis_panel is None:
            self._frequency_analysis_panel = FrequencyAnalysisDialog(
                self._build_frequency_analysis_input(),
                visible_range_provider=self.plot_widget.visible_time_range,
                show_controls_inline=False,
                parent=self.analysis_tab,
            )
            self.analysis_tab_layout.removeWidget(self.analysis_placeholder)
            self.analysis_placeholder.hide()
            self.analysis_tab_layout.addWidget(self._frequency_analysis_panel)
            self.analysis_sidebar_layout.insertWidget(
                0, self._frequency_analysis_panel.analysis_controls_widget()
            )
        return self._frequency_analysis_panel

    def _sync_sidebar_to_current_tab(self, index: int) -> None:
        if (
            index == self.analysis_tab_index
            and self.content_tabs.isTabEnabled(self.analysis_tab_index)
        ):
            self._ensure_frequency_analysis_panel()
            self.sidebar_stack.setCurrentWidget(self.analysis_sidebar)
            return
        self.sidebar_stack.setCurrentIndex(0)

    def _build_frequency_analysis_input(self) -> FrequencyAnalysisInput:
        if self.current_record is None:
            raise RuntimeError("Frequency analysis requires an active record.")
        return FrequencyAnalysisInput(
            record=self.current_record,
            processed_signal=self.processed_signal,
            filtered_available=self.filter_config.any_enabled()
            and self.processed_signal is not None,
            active_lead_index=self.controls.primary_selected_lead_index(),
        )
