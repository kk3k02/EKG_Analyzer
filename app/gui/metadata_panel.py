from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLabel, QWidget

from app.models.ecg_record import ECGRecord
from app.utils.time_utils import format_seconds


class MetadataPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._labels: dict[str, QLabel] = {}
        layout = QFormLayout(self)

        for key in (
            "Plik",
            "Format",
            "Czestotliwosc probkowania",
            "Probki",
            "Odprowadzenia",
            "Czas trwania",
            "Jednostki",
        ):
            label = QLabel("-")
            label.setTextInteractionFlags(label.textInteractionFlags())
            self._labels[key] = label
            layout.addRow(f"{key}:", label)

    def set_record(self, record: ECGRecord | None) -> None:
        if record is None:
            for label in self._labels.values():
                label.setText("-")
            return

        self._labels["Plik"].setText(record.file_name)
        self._labels["Format"].setText(record.source_format.upper())
        self._labels["Czestotliwosc probkowania"].setText(f"{record.sampling_rate:.2f} Hz")
        self._labels["Probki"].setText(str(record.n_samples))
        self._labels["Odprowadzenia"].setText(str(record.n_leads))
        self._labels["Czas trwania"].setText(format_seconds(record.duration_seconds))
        self._labels["Jednostki"].setText(record.units)
