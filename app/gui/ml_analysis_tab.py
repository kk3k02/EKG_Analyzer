from __future__ import annotations
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from app.models.ecg_record import ECGRecord
from app.services.ml_analysis import MLModel, load_model, predict_signal


class MLAnalysisSignals(QObject):
    """Custom signals for the ML analysis background task."""

    finished = Signal(dict)
    failed = Signal(str)
    progress = Signal(str)


class MLAnalysisTask(QRunnable):
    """Background task to run ML inference without freezing the GUI."""

    def __init__(self, signal: np.ndarray, model_dir: Path):
        super().__init__()
        self.signals = MLAnalysisSignals()
        self._signal = signal
        self._model_dir = model_dir

    def run(self) -> None:
        try:
            self.signals.progress.emit("Wczytywanie modelu...")
            model = load_model(self._model_dir)

            self.signals.progress.emit("Przetwarzanie sygnału i predykcja...")
            results = predict_signal(self._signal, model, return_windows=True)

            self.signals.finished.emit(results)
        except Exception as e:
            self.signals.failed.emit(str(e))


class MLAnalysisTab(QWidget):
    """Widget for the Machine Learning analysis tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._record: ECGRecord | None = None
        self._main_window = parent

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left side - Controls
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_container.setMinimumWidth(320)
        controls_container.setMaximumWidth(380)

        # Right side - Results
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)

        main_layout.addWidget(controls_container)
        main_layout.addWidget(results_container, stretch=1)

        # Controls Group
        controls_group = QGroupBox("Panel sterowania", self)
        form_layout = QFormLayout(controls_group)

        self.lead_combo = QComboBox(self)
        self.run_button = QPushButton("Uruchom analizę", self)
        self.status_label = QLabel("Gotowy.", self)
        self.status_label.setWordWrap(True)

        form_layout.addRow("Odprowadzenie:", self.lead_combo)
        form_layout.addRow(self.run_button)
        form_layout.addRow("Status:", self.status_label)

        controls_layout.addWidget(controls_group)

        # Results Group
        results_group = QGroupBox("Wyniki analizy", self)
        self.results_layout = QFormLayout(results_group)
        self.results_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        self.majority_class_label = QLabel("-", self)
        self.majority_prob_label = QLabel("-", self)
        self.distribution_label = QLabel("-", self)
        self.n_windows_label = QLabel("-", self)

        self.results_layout.addRow("Dominująca klasa:", self.majority_class_label)
        self.results_layout.addRow("Średnia pewność:", self.majority_prob_label)
        self.results_layout.addRow("Liczba okien:", self.n_windows_label)
        self.results_layout.addRow("Rozkład klas (%):", self.distribution_label)

        # Detailed results table
        self.details_table = QTableWidget(self)
        self.details_table.setColumnCount(4)
        self.details_table.setHorizontalHeaderLabels(
            ["Start (s)", "Koniec (s)", "Klasa", "Pewność"]
        )
        self.details_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.details_table.setVisible(True)  # Show by default

        results_layout.addWidget(results_group)
        results_layout.addWidget(self.details_table)

        # Connect signals
        self.run_button.clicked.connect(self.run_analysis)

        controls_layout.addWidget(controls_group)

        # Legend Group
        legend_group = QGroupBox("Legenda klas", self)
        legend_layout = QVBoxLayout(legend_group)

        legend_table = QTableWidget(3, 2, self)
        legend_table.setHorizontalHeaderLabels(["Klasa", "Opis"])
        legend_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        legend_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        legend_table.verticalHeader().setVisible(False)
        legend_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        legend_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        legend_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        _LEGEND = [
            ("ARR", "Arytmia (Arrhythmia)"),
            ("CHF", "Zastoinowa niewydolność serca\n(Congestive Heart Failure)"),
            ("NSR", "Prawidłowy rytm zatokowy\n(Normal Sinus Rhythm)"),
        ]
        for row, (cls, desc) in enumerate(_LEGEND):
            legend_table.setItem(row, 0, QTableWidgetItem(cls))
            legend_table.setItem(row, 1, QTableWidgetItem(desc))

        legend_table.resizeRowsToContents()
        legend_layout.addWidget(legend_table)

        controls_layout.addWidget(legend_group)
        controls_layout.addStretch(1)

    def update_record(self, record: ECGRecord | None) -> None:
        self._record = record
        self.lead_combo.clear()
        if record:
            self.lead_combo.addItems(record.lead_names)
            self.run_button.setEnabled(True)
        else:
            self.run_button.setEnabled(False)
        self._clear_results()

    def run_analysis(self) -> None:
        if self._record is None:
            QMessageBox.warning(self, "Brak danych", "Najpierw wczytaj plik EKG.")
            return

        self.run_button.setEnabled(False)
        self.status_label.setText("Inicjalizacja...")
        self._clear_results()
        QApplication.processEvents()

        lead_index = self.lead_combo.currentIndex()
        signal_1d = self._record.signal[:, lead_index]

        # Path to models directory
        model_dir = Path(__file__).resolve().parents[2] / "app" / "models" / "ml"

        task = MLAnalysisTask(signal=signal_1d, model_dir=model_dir)
        task.signals.finished.connect(self._handle_results)
        task.signals.failed.connect(self._handle_error)
        task.signals.progress.connect(self.status_label.setText)

        self._main_window.thread_pool.start(task)

    def _clear_results(self) -> None:
        self.majority_class_label.setText("-")
        self.majority_prob_label.setText("-")
        self.distribution_label.setText("-")
        self.n_windows_label.setText("-")
        self.details_table.setRowCount(0)

    def _handle_results(self, results: dict) -> None:
        self.status_label.setText("Analiza zakończona.")
        self.run_button.setEnabled(True)

        self.majority_class_label.setText(
            f"<b>{results.get('majority_class', '-')}</b>"
        )
        self.majority_prob_label.setText(
            f"{results.get('majority_prob', 0) * 100:.1f}%"
        )
        self.n_windows_label.setText(str(results.get("n_windows", 0)))

        dist = results.get("class_distribution", {})
        dist_str = ", ".join([f"{k}: {v}%" for k, v in dist.items()])
        self.distribution_label.setText(dist_str)

        window_results = results.get("window_results", [])
        self.details_table.setRowCount(len(window_results))
        for i, row in enumerate(window_results):
            self.details_table.setItem(i, 0, QTableWidgetItem(f"{row['start_s']:.2f}"))
            self.details_table.setItem(i, 1, QTableWidgetItem(f"{row['end_s']:.2f}"))
            self.details_table.setItem(i, 2, QTableWidgetItem(row["class"]))

            probs_str = ", ".join([f"{k}: {v:.2f}" for k, v in row["probs"].items()])
            self.details_table.setItem(i, 3, QTableWidgetItem(probs_str))

        self.details_table.resizeRowsToContents()

    def _handle_error(self, message: str) -> None:
        self.status_label.setText(f"Błąd: {message}")
        self.run_button.setEnabled(True)
        QMessageBox.critical(self, "Błąd analizy ML", message)
