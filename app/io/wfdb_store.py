from __future__ import annotations

from pathlib import Path

import numpy as np

from app.io.base_store import BaseECGStore
from app.models.ecg_record import ECGRecord
from app.services.validation import build_time_axis
from app.utils.file_utils import normalize_path


class WFDBECGStore(BaseECGStore):
    source_format = "wfdb"
    display_name = "WFDB"
    load_extensions = (".hea", ".dat", ".atr")
    save_extensions = (".hea",)
    default_save_extension = ".hea"

    def load(self, file_path: str) -> ECGRecord:
        import wfdb

        normalized_path = Path(normalize_path(file_path))
        record_name = normalized_path.with_suffix("")

        if normalized_path.suffix.lower() == ".dat":
            header_path = record_name.with_suffix(".hea")
            if not header_path.exists():
                raise FileNotFoundError("WFDB header file (.hea) is missing for the selected .dat file.")
        elif normalized_path.suffix.lower() == ".hea":
            data_path = record_name.with_suffix(".dat")
            if not data_path.exists():
                raise FileNotFoundError("WFDB data file (.dat) is missing for the selected record.")

        record = wfdb.rdrecord(str(record_name))
        signal = np.asarray(record.p_signal if record.p_signal is not None else record.d_signal, dtype=float)
        if signal.ndim == 1:
            signal = signal[:, np.newaxis]

        sampling_rate = float(record.fs)
        time_axis = build_time_axis(signal.shape[0], sampling_rate)

        annotations = []
        ann_path = record_name.with_suffix(".atr")
        if ann_path.exists():
            ann = wfdb.rdann(str(record_name), "atr")
            for i in range(len(ann.sample)):
                annotations.append({
                    "sample": int(ann.sample[i]),
                    "time": float(ann.sample[i] / sampling_rate),
                    "symbol": str(ann.symbol[i]) if ann.symbol else "",
                    "label": str(ann.aux_note[i]) if ann.aux_note else ""
                })
            

        metadata = {
            "record_name": record.record_name,
            "base_time": str(record.base_time) if record.base_time else None,
            "base_date": str(record.base_date) if record.base_date else None,
            "comments": list(record.comments or []),
            "sig_len": int(record.sig_len),
        }

        return ECGRecord(
            source_format="wfdb",
            file_path=str(normalized_path),
            sampling_rate=sampling_rate,
            lead_names=list(record.sig_name),
            signal=signal,
            time_axis=time_axis,
            units="mV",
            metadata=metadata,
            annotations=annotations,
        )

    def save(self, record: ECGRecord, file_path: str) -> str:
        import wfdb

        normalized_path = normalize_path(self.ensure_save_extension(file_path))
        output_path = Path(normalized_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        record_name = output_path.stem

        comments = [str(comment) for comment in record.metadata.get("comments", []) if str(comment).strip()]
        wfdb.wrsamp(
            record_name=record_name,
            fs=float(record.sampling_rate),
            units=[record.units] * record.n_leads,
            sig_name=record.lead_names,
            p_signal=np.asarray(record.signal, dtype=float),
            comments=comments,
            write_dir=str(output_path.parent),
        )

        self._write_annotations(record, record_name=record_name, write_dir=str(output_path.parent))
        return str(output_path)

    def _write_annotations(self, record: ECGRecord, *, record_name: str, write_dir: str) -> None:
        import wfdb

        valid_annotations = [
            annotation
            for annotation in record.annotations
            if isinstance(annotation.get("sample"), (int, np.integer))
            and int(annotation["sample"]) >= 0
            and str(annotation.get("symbol", "")).strip()
        ]
        if not valid_annotations:
            return

        wfdb.wrann(
            record_name=record_name,
            extension="atr",
            sample=np.asarray([int(annotation["sample"]) for annotation in valid_annotations], dtype=int),
            symbol=[str(annotation.get("symbol", "")).strip() for annotation in valid_annotations],
            aux_note=[str(annotation.get("label", "") or "") for annotation in valid_annotations],
            fs=float(record.sampling_rate),
            write_dir=write_dir,
        )
