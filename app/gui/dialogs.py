from __future__ import annotations

from PySide6.QtCore import QEasingCurve, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.metadata_panel import MetadataPanel
from app.models.ecg_record import ECGRecord


class BusyIndicator(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(24)
        self._offset = 0.0
        self._animation = QVariantAnimation(self)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.setDuration(900)
        self._animation.setLoopCount(-1)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._animation.valueChanged.connect(self._on_value_changed)

    def start(self) -> None:
        if self._animation.state() != QVariantAnimation.State.Running:
            self._animation.start()

    def stop(self) -> None:
        if self._animation.state() == QVariantAnimation.State.Running:
            self._animation.stop()
        self._offset = 0.0
        self.update()

    def _on_value_changed(self, value) -> None:
        self._offset = float(value)
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        track_rect = self.rect().adjusted(2, 4, -2, -4)
        painter.setPen(QColor("#000000"))
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawRoundedRect(track_rect, 8, 8)

        travel = max(track_rect.width() - 72, 1)
        x_pos = track_rect.x() + int(travel * self._offset)
        chunk_rect = track_rect.adjusted(3, 3, -(track_rect.width() - 72) - 3, -3)
        chunk_rect.moveLeft(x_pos)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#D10000"))
        painter.drawRoundedRect(chunk_rect, 6, 6)


class WaitPopupDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.CustomizeWindowHint
        )
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setModal(True)
        self.setFixedSize(340, 128)
        self.setStyleSheet(
            "QDialog {"
            "background-color: #FFF4F4;"
            "border: 4px solid #000000;"
            "border-radius: 10px;"
            "}"
            "QLabel {"
            "color: #D10000;"
            "font-size: 24px;"
            "font-weight: 800;"
            "background: transparent;"
            "border: none;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        self.message_label = QLabel("prosze czekac", self)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message_label)

        self.indicator = BusyIndicator(self)
        layout.addWidget(self.indicator)

    def showEvent(self, event) -> None:
        self.indicator.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self.indicator.stop()
        super().hideEvent(event)


class SamplingRateDialog(QDialog):
    def __init__(self, initial_value: float = 250.0, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sampling rate")

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("CSV/TXT file has no explicit time axis. Confirm the sampling rate.")
        )

        self.spin_box = QDoubleSpinBox(self)
        self.spin_box.setRange(0.1, 10000.0)
        self.spin_box.setDecimals(2)
        self.spin_box.setValue(initial_value)
        self.spin_box.setSuffix(" Hz")

        form = QFormLayout()
        form.addRow("Sampling rate:", self.spin_box)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def sampling_rate(self) -> float:
        return float(self.spin_box.value())


class MetadataDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Informacje o pliku")
        self.setModal(True)
        self.resize(420, 260)

        layout = QVBoxLayout(self)
        self.metadata_panel = MetadataPanel(self)
        layout.addWidget(self.metadata_panel)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_button = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.setText("Zamknij")
            close_button.clicked.connect(self.close)
        layout.addWidget(buttons)

    def set_record(self, record: ECGRecord | None) -> None:
        self.metadata_panel.set_record(record)


class DiseaseResultDialog(QDialog):
    def __init__(self, result: dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("Wynik analizy schorzen EKG")
        self.setModal(True)
        self.resize(560, 480)

        layout = QVBoxLayout(self)

        title = QLabel("<span>&#9829;</span> Wynik analizy schorzen EKG", self)
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)

        class_info = result["class_info"]
        predicted_class = result["predicted_class"]
        result_card = QLabel(self)
        result_card.setWordWrap(True)
        result_card.setStyleSheet(
            "QLabel {"
            f"background-color: {class_info['color']};"
            "color: white;"
            "border-radius: 10px;"
            "padding: 14px;"
            "font-size: 14px;"
            "}"
        )
        result_card.setText(
            f"<b>{class_info['name']} ({predicted_class})</b><br>{class_info['description']}"
        )
        layout.addWidget(result_card)

        layout.addWidget(QLabel("Pewnosc predykcji", self))
        confidence_bar = QProgressBar(self)
        confidence_bar.setRange(0, 100)
        confidence_bar.setValue(int(round(float(result["confidence"]) * 100.0)))
        confidence_bar.setFormat("%p%")
        layout.addWidget(confidence_bar)

        table = QTableWidget(3, 3, self)
        table.setHorizontalHeaderLabels(["Klasa", "Glosy", "Prawdopodobienstwo"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        votes = result["votes"]
        probabilities = result["probabilities"]
        for row, class_name in enumerate(("ARR", "CHF", "NSR")):
            table.setItem(row, 0, QTableWidgetItem(class_name))
            table.setItem(row, 1, QTableWidgetItem(str(votes[class_name])))
            table.setItem(
                row,
                2,
                QTableWidgetItem(f"{float(probabilities[class_name]) * 100.0:.1f}%"),
            )
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(table)

        technical_box = QWidget(self)
        technical_layout = QVBoxLayout(technical_box)
        technical_layout.setContentsMargins(0, 0, 0, 0)
        technical_layout.addWidget(
            QLabel(f"Przeanalizowane segmenty: {result['n_segments']}", technical_box)
        )
        technical_layout.addWidget(
            QLabel(
                f"Uzyte modele: {', '.join(result['models_used'])}",
                technical_box,
            )
        )
        layout.addWidget(technical_box)

        warning = QLabel(
            "Wynik ma charakter pomocniczy. Skonsultuj sie z lekarzem.", self
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #b00020; font-weight: 600;")
        layout.addWidget(warning)

        buttons_row = QHBoxLayout()
        self.copy_button = QPushButton("Kopiuj wynik do schowka", self)
        self.copy_button.clicked.connect(self._copy_result_to_clipboard)
        buttons_row.addWidget(self.copy_button)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        close_button = button_box.button(QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.setText("Zamknij")
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self.accept)
        buttons_row.addWidget(button_box)
        layout.addLayout(buttons_row)

    def _copy_result_to_clipboard(self) -> None:
        class_info = self.result["class_info"]
        text = (
            "Wynik analizy schorzen EKG\n"
            f"Klasa: {self.result['predicted_class']} - {class_info['name']}\n"
            f"Opis: {class_info['description']}\n"
            f"Pewnosc: {float(self.result['confidence']) * 100.0:.1f}%\n"
            f"Glosy: {self.result['votes']}\n"
            f"Prawdopodobienstwa: {self.result['probabilities']}\n"
            f"Modele: {', '.join(self.result['models_used'])}\n"
            f"Segmenty: {self.result['n_segments']}"
        )
        QApplication.clipboard().setText(text)
